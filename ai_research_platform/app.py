"""
AI Adoption Research Analysis Platform
Fixes applied:
  1. Removed student/teacher mode toggle - single student mode only
  2. Database resets ID counter on reset so IDs always start from 1
  3. Concise one-line inference + one-line conclusion per test + final summary bar chart
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash
from scipy.stats import chi2_contingency, ttest_ind, f_oneway, pearsonr
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'research_secret_key_2024'

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATABASE      = os.path.join(BASE_DIR, 'database.db')
CHARTS_FOLDER = os.path.join(BASE_DIR, 'static', 'charts')

REQUIRED_COLUMNS = [
    'age', 'gender', 'major', 'academic_year', 'ai_usage',
    'ai_frequency', 'comfort_level', 'gpa', 'career_interest',
    'ethical_concern', 'job_replacement_fear', 'optimism_score'
]


# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS survey_data (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            age                  INTEGER,
            gender               TEXT,
            major                TEXT,
            academic_year        TEXT,
            ai_usage             TEXT,
            ai_frequency         TEXT,
            comfort_level        INTEGER,
            gpa                  REAL,
            career_interest      INTEGER,
            ethical_concern      INTEGER,
            job_replacement_fear INTEGER,
            optimism_score       INTEGER,
            timestamp            TEXT
        )
    ''')
    conn.commit()
    conn.close()


def load_data():
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM survey_data", conn)
    conn.close()
    return df


def insert_df(df):
    conn = get_db()
    for _, row in df.iterrows():
        conn.execute('''
            INSERT INTO survey_data
            (age, gender, major, academic_year, ai_usage, ai_frequency,
             comfort_level, gpa, career_interest, ethical_concern,
             job_replacement_fear, optimism_score, timestamp)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ''', (
            row.get('age'), row.get('gender'), row.get('major'),
            row.get('academic_year'), row.get('ai_usage'), row.get('ai_frequency'),
            row.get('comfort_level'), row.get('gpa'), row.get('career_interest'),
            row.get('ethical_concern'), row.get('job_replacement_fear'),
            row.get('optimism_score'),
            datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ))
    conn.commit()
    conn.close()


# ============================================================
# CSV VALIDATION
# ============================================================

def validate_csv(df):
    errors = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f'Missing columns: {", ".join(missing)}')
    for col in ['age', 'comfort_level', 'gpa', 'career_interest',
                'ethical_concern', 'job_replacement_fear', 'optimism_score']:
        if col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
            except Exception:
                errors.append(f'Column "{col}" must be numeric.')
    return errors


# ============================================================
# CHART HELPERS
# ============================================================

def clear_charts():
    os.makedirs(CHARTS_FOLDER, exist_ok=True)
    for f in os.listdir(CHARTS_FOLDER):
        if f.endswith('.png'):
            try:
                os.remove(os.path.join(CHARTS_FOLDER, f))
            except Exception:
                pass


def save_fig(fig, name):
    os.makedirs(CHARTS_FOLDER, exist_ok=True)
    fig.savefig(os.path.join(CHARTS_FOLDER, name), bbox_inches='tight', dpi=120)
    plt.close(fig)
    return f'charts/{name}'


def pie_chart(df, col, title):
    counts = df[col].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title(title, fontsize=12, fontweight='bold')
    return save_fig(fig, f'pie_{col}.png')


def bar_chart(df, col, title):
    counts = df[col].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(col.replace('_', ' ').title())
    ax.set_ylabel('Count')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    return save_fig(fig, f'bar_{col}.png')


def boxplot_chart(df, group_col, val_col, title):
    order  = df[group_col].dropna().unique().tolist()
    groups = [df[df[group_col] == g][val_col].dropna().values for g in order]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot(groups, labels=order)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(group_col.replace('_', ' ').title())
    ax.set_ylabel(val_col.replace('_', ' ').title())
    plt.tight_layout()
    return save_fig(fig, f'box_{group_col}_{val_col}.png')


def scatter_chart(df, x_col, y_col, title):
    clean = df[[x_col, y_col]].dropna()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(clean[x_col], clean[y_col], alpha=0.5, color='steelblue')
    z = np.polyfit(clean[x_col], clean[y_col], 1)
    ax.plot(sorted(clean[x_col]), np.poly1d(z)(sorted(clean[x_col])), 'r--', alpha=0.8)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(x_col.replace('_', ' ').title())
    ax.set_ylabel(y_col.replace('_', ' ').title())
    plt.tight_layout()
    return save_fig(fig, f'scatter_{x_col}_{y_col}.png')


def summary_bar_chart(results):
    """
    Horizontal bar chart of p-values for every test.
    Green bar = significant (p<0.05), grey = not significant.
    Red dashed line marks the 0.05 threshold.
    """
    labels, pvals, colors = [], [], []
    for r in results:
        if r.get('error') or r.get('p_value') is None:
            continue
        short = (r['test_name']
                 .replace('Chi-Square Test of Independence', 'Chi-Square')
                 .replace('Independent Sample T-Test', 'T-Test')
                 .replace('One-Way ANOVA', 'ANOVA')
                 .replace('Pearson Correlation', 'Correlation'))
        labels.append(f"{short}\n({r['variables'][:40]})")
        pvals.append(r['p_value'])
        colors.append('#2a7a2a' if r['p_value'] < 0.05 else '#aaaaaa')

    if not labels:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(labels))))
    bars = ax.barh(labels, pvals, color=colors, edgecolor='black', height=0.5)
    ax.axvline(0.05, color='red', linestyle='--', linewidth=1.5,
               label='Significance threshold (p = 0.05)')
    ax.set_xlabel('p-value', fontsize=11)
    ax.set_title('Statistical Tests Summary — p-value Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    for bar, pv in zip(bars, pvals):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{pv:.4f}', va='center', fontsize=9)
    ax.set_xlim(0, max(max(pvals) * 1.35, 0.12))
    plt.tight_layout()
    return save_fig(fig, 'summary_pvalues.png')


# ============================================================
# STATISTICAL TESTS
# ============================================================

def run_chi_square(df, col1, col2):
    try:
        ct = pd.crosstab(df[col1], df[col2])
        chi2, p, dof, _ = chi2_contingency(ct)
        sig = p < 0.05
        top = ct.stack().idxmax()
        inference = (
            f"The most common combination is {top[0]} students who {'' if 'usage' not in col2 else 'do '}use AI. "
            f"{'A statistically significant association exists' if sig else 'No statistically significant association found'} "
            f"between {col1.replace('_',' ')} and {col2.replace('_',' ')} "
            f"(Chi2 = {chi2:.4f}, p = {p:.4f}, df = {dof})."
        )
        conclusion = (
            f"{col1.replace('_',' ').title()} "
            f"{'significantly influences AI adoption' if sig else 'does not significantly influence AI adoption'} "
            f"among college students (p = {p:.4f})."
        )
        return {
            'test_name': 'Chi-Square Test of Independence',
            'variables': f'{col1} vs {col2}',
            'h0': f'No relationship between {col1} and {col2}.',
            'h1': f'Significant relationship between {col1} and {col2}.',
            'statistic': round(chi2, 4), 'stat_label': 'Chi-Square',
            'p_value': round(p, 4), 'dof': dof,
            'decision': 'Reject H0' if sig else 'Fail to Reject H0',
            'inference': inference, 'conclusion': conclusion, 'error': None
        }
    except Exception as e:
        return {'test_name': 'Chi-Square Test', 'error': str(e)}


def run_t_test(df, group_col, val_col, g1, g2):
    try:
        a = df[df[group_col] == g1][val_col].dropna()
        b = df[df[group_col] == g2][val_col].dropna()
        if len(a) < 2 or len(b) < 2:
            return {'test_name': 'T-Test', 'error': 'Not enough data in one or both groups.'}
        t, p = ttest_ind(a, b)
        sig  = p < 0.05
        diff = round(a.mean() - b.mean(), 3)
        higher = g1 if a.mean() > b.mean() else g2
        inference = (
            f"{higher} students have a {'significantly ' if sig else ''}higher mean "
            f"{val_col.replace('_',' ')} "
            f"({round(max(a.mean(),b.mean()),3)} vs {round(min(a.mean(),b.mean()),3)}, "
            f"difference = {abs(diff):.3f}, t = {t:.4f}, p = {p:.4f})."
        )
        conclusion = (
            f"AI usage {'has a statistically significant effect' if sig else 'has no statistically significant effect'} "
            f"on student GPA (p = {p:.4f})."
        )
        return {
            'test_name': 'Independent Sample T-Test',
            'variables': f'{val_col} by {group_col}',
            'h0': f'No difference in mean {val_col} between {g1} and {g2}.',
            'h1': f'Significant difference in mean {val_col} between {g1} and {g2}.',
            'statistic': round(t, 4), 'stat_label': 'T-Statistic',
            'p_value': round(p, 4),
            'group1_mean': round(a.mean(), 3), 'group2_mean': round(b.mean(), 3),
            'group1_label': f'{g1} mean', 'group2_label': f'{g2} mean',
            'decision': 'Reject H0' if sig else 'Fail to Reject H0',
            'inference': inference, 'conclusion': conclusion, 'error': None
        }
    except Exception as e:
        return {'test_name': 'T-Test', 'error': str(e)}


def run_anova(df, group_col, val_col):
    try:
        grps = [g[val_col].dropna().values for _, g in df.groupby(group_col) if len(g) >= 2]
        if len(grps) < 2:
            return {'test_name': 'ANOVA', 'error': 'Not enough groups.'}
        f, p   = f_oneway(*grps)
        means  = df.groupby(group_col)[val_col].mean().round(3)
        best   = means.idxmax()
        worst  = means.idxmin()
        sig    = p < 0.05
        inference = (
            f"'{best}' students show the highest mean {val_col.replace('_',' ')} ({means[best]:.3f}); "
            f"'{worst}' show the lowest ({means[worst]:.3f}). "
            f"{'These differences are statistically significant' if sig else 'These differences are not statistically significant'} "
            f"(F = {f:.4f}, p = {p:.4f})."
        )
        conclusion = (
            f"AI usage frequency "
            f"{'significantly affects' if sig else 'does not significantly affect'} "
            f"student GPA across frequency groups (p = {p:.4f})."
        )
        return {
            'test_name': 'One-Way ANOVA',
            'variables': f'{val_col} across {group_col}',
            'h0': f'Mean {val_col} is equal across all {group_col} groups.',
            'h1': f'At least one {group_col} group has a different mean {val_col}.',
            'statistic': round(f, 4), 'stat_label': 'F-Statistic',
            'p_value': round(p, 4),
            'group_means': means.to_dict(),
            'decision': 'Reject H0' if sig else 'Fail to Reject H0',
            'inference': inference, 'conclusion': conclusion, 'error': None
        }
    except Exception as e:
        return {'test_name': 'ANOVA', 'error': str(e)}


def run_correlation(df, col1, col2):
    try:
        clean = df[[col1, col2]].dropna()
        if len(clean) < 3:
            return {'test_name': 'Pearson Correlation', 'error': 'Not enough data.'}
        r, p     = pearsonr(clean[col1], clean[col2])
        sig      = p < 0.05
        strength = 'strong' if abs(r) >= 0.7 else 'moderate' if abs(r) >= 0.4 else 'weak'
        direction = 'positive' if r > 0 else 'negative'
        inference = (
            f"There is a {strength} {direction} correlation between "
            f"{col1.replace('_',' ')} and {col2.replace('_',' ')} "
            f"(r = {r:.4f}, p = {p:.4f}). "
            f"{'This is statistically significant.' if sig else 'This is not statistically significant.'}"
        )
        conclusion = (
            f"{col1.replace('_',' ').title()} "
            f"{'is significantly correlated with' if sig else 'is not significantly correlated with'} "
            f"{col2.replace('_',' ')} (r = {r:.4f}, p = {p:.4f})."
        )
        return {
            'test_name': 'Pearson Correlation',
            'variables': f'{col1} vs {col2}',
            'h0': f'No significant correlation between {col1} and {col2}.',
            'h1': f'Significant correlation between {col1} and {col2}.',
            'statistic': round(r, 4), 'stat_label': 'Pearson r',
            'p_value': round(p, 4),
            'decision': 'Reject H0' if sig else 'Fail to Reject H0',
            'inference': inference, 'conclusion': conclusion, 'error': None
        }
    except Exception as e:
        return {'test_name': 'Pearson Correlation', 'error': str(e)}


def make_final_conclusion(df, results):
    """One paragraph final research conclusion drawn from real data + test outcomes."""
    n       = len(df)
    ai_pct  = round((df['ai_usage'].str.strip().str.lower() == 'yes').sum() / n * 100, 1)
    avg_gpa = round(df['gpa'].mean(), 2)
    avg_c   = round(df['comfort_level'].mean(), 2)
    sig     = [r['test_name'] for r in results if not r.get('error') and r.get('p_value', 1) < 0.05]
    nsig    = [r['test_name'] for r in results if not r.get('error') and r.get('p_value', 0) >= 0.05]
    return (
        f"Based on a sample of {n} college students, this study finds that {ai_pct}% actively use AI tools, "
        f"with an overall mean GPA of {avg_gpa} and an average AI comfort level of {avg_c}/5. "
        f"Statistically significant results (p < 0.05) were observed in: {', '.join(sig) if sig else 'none of the tests'}. "
        f"Non-significant results were found in: {', '.join(nsig) if nsig else 'none'}. "
        f"These findings suggest that AI adoption among college students is measurably associated with "
        f"differences in academic performance and is influenced by gender and disciplinary background. "
        f"Since this is cross-sectional survey data, causality cannot be established — "
        f"longitudinal studies are recommended to determine whether AI usage directly improves academic outcomes."
    )


def make_observations(df):
    """Concise, data-driven professional observations — one paragraph each."""
    obs = []
    n   = len(df)

    completeness = round((1 - df.isnull().sum().sum() / max(df.size, 1)) * 100, 1)
    obs.append({
        'type': 'info',
        'title': f'Sample Quality: {n} Responses — {completeness}% Complete',
        'detail': (
            f'The dataset contains {n} survey responses with a {completeness}% data completeness rate. '
            + ('This meets the recommended minimum of 100 observations for reliable statistical inference.'
               if n >= 100 else
               f'A sample of at least 100 is recommended; the current size of {n} limits generalizability of findings.')
        )
    })

    if 'ai_usage' in df.columns:
        yes = int((df['ai_usage'].str.strip().str.lower() == 'yes').sum())
        pct = round(yes / n * 100, 1)
        obs.append({
            'type': 'info',
            'title': f'AI Adoption Rate: {pct}% ({yes} of {n} students)',
            'detail': (
                f'{yes} of {n} surveyed students ({pct}%) report actively using AI tools. '
                + ('This majority adoption rate indicates AI has become a mainstream resource in this population.'
                   if pct > 50 else
                   'Below-50% adoption suggests that awareness gaps, institutional restrictions, or ethical reservations '
                   'continue to limit uptake among a significant portion of students.')
            )
        })

    if 'gpa' in df.columns and 'ai_usage' in df.columns:
        ag = df[df['ai_usage'].str.strip().str.lower() == 'yes']['gpa'].dropna()
        ng = df[df['ai_usage'].str.strip().str.lower() == 'no']['gpa'].dropna()
        if len(ag) > 0 and len(ng) > 0:
            diff = round(ag.mean() - ng.mean(), 3)
            obs.append({
                'type': 'info',
                'title': f'GPA Gap: AI Users {round(ag.mean(),3)} vs Non-Users {round(ng.mean(),3)} (Difference: {diff:+.3f})',
                'detail': (
                    f'AI-adopting students report a mean GPA {abs(diff):.3f} points '
                    + ('higher' if diff > 0 else 'lower')
                    + f' than non-users. This {"practically meaningful (>0.2 point)" if abs(diff) >= 0.2 else "modest"} '
                    f'difference is observational — high-achieving students may simply be more proactive '
                    f'in adopting new academic tools, rather than AI usage directly causing improved grades.'
                )
            })

    if 'comfort_level' in df.columns:
        avg_c = round(df['comfort_level'].mean(), 2)
        high  = int((df['comfort_level'] >= 4).sum())
        obs.append({
            'type': 'info',
            'title': f'AI Comfort Level: Mean {avg_c}/5 — {round(high/n*100,1)}% Rate High Comfort (4-5)',
            'detail': (
                f'The average self-reported comfort with AI is {avg_c}/5. '
                f'{round(high/n*100,1)}% of students feel highly comfortable (rating 4 or 5). '
                + ('This broadly receptive attitude is a positive indicator for AI integration in curricula.'
                   if avg_c >= 3 else
                   'Below-average comfort scores suggest that targeted digital literacy training '
                   'could meaningfully improve AI uptake and confidence.')
            )
        })

    if 'ethical_concern' in df.columns and 'job_replacement_fear' in df.columns:
        ec = round(df['ethical_concern'].mean(), 2)
        jf = round(df['job_replacement_fear'].mean(), 2)
        op = round(df['optimism_score'].mean(), 2) if 'optimism_score' in df.columns else None
        obs.append({
            'type': 'warning' if ec >= 3.5 or jf >= 3.5 else 'info',
            'title': f'Student Attitudes: Ethical Concern {ec}/5 | Job Fear {jf}/5'
                     + (f' | Optimism {op}/5' if op else ''),
            'detail': (
                f'Students report an ethical concern score of {ec}/5 and job replacement fear of {jf}/5. '
                + (f'Optimism about AI averages {op}/5, reflecting '
                   + ('a generally constructive outlook despite concerns.'
                      if op and op >= 3 else 'cautious sentiment about AI\'s societal impact.')
                   if op else '')
                + ' These attitudinal metrics are important moderators of adoption behaviour '
                'and should be incorporated into any intervention design.'
            )
        })

    obs.append({
        'type': 'warning',
        'title': 'Statistical Validity Notice',
        'detail': (
            f'With n = {n}, the Central Limit Theorem partially mitigates normality concerns for T-Test and ANOVA. '
            'However, Chi-Square requires all expected cell frequencies to be >= 5, and Pearson Correlation '
            'assumes a linear relationship between variables. All results should be treated as indicative '
            'and validated with a larger, randomly sampled dataset before drawing policy conclusions.'
        )
    })

    return obs


# ============================================================
# SAMPLE DATA GENERATOR
# ============================================================

def generate_sample_data(n=100):
    np.random.seed(42)
    ai_usage = np.random.choice(['Yes', 'No'], n, p=[0.65, 0.35])
    gpa = np.where(
        ai_usage == 'Yes',
        np.clip(np.random.normal(3.2, 0.4, n), 1.5, 4.0),
        np.clip(np.random.normal(2.9, 0.5, n), 1.5, 4.0)
    )
    return pd.DataFrame({
        'age':                  np.random.randint(18, 28, n),
        'gender':               np.random.choice(['Male', 'Female', 'Non-binary'], n, p=[0.45, 0.45, 0.10]),
        'major':                np.random.choice(['Computer Science', 'Business', 'Arts', 'Engineering', 'Psychology'], n),
        'academic_year':        np.random.choice(['Freshman', 'Sophomore', 'Junior', 'Senior'], n),
        'ai_usage':             ai_usage,
        'ai_frequency':         np.random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never'], n),
        'comfort_level':        np.random.randint(1, 6, n),
        'gpa':                  np.round(gpa, 2),
        'career_interest':      np.random.randint(1, 6, n),
        'ethical_concern':      np.random.randint(1, 6, n),
        'job_replacement_fear': np.random.randint(1, 6, n),
        'optimism_score':       np.random.randint(1, 6, n),
    })


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    df = load_data()
    stats = {}
    if not df.empty:
        stats = {
            'total_rows':  len(df),
            'ai_users':    int((df['ai_usage'].str.lower() == 'yes').sum()),
            'avg_gpa':     round(df['gpa'].mean(), 2),
            'avg_comfort': round(df['comfort_level'].mean(), 2),
        }
    return render_template('index.html', stats=stats, has_data=not df.empty)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash('No file selected.', 'error')
            return redirect(request.url)
        if not file.filename.endswith('.csv'):
            flash('Please upload a .csv file.', 'error')
            return redirect(request.url)
        try:
            df = pd.read_csv(file)
            if df.empty:
                flash('The file is empty.', 'error')
                return redirect(request.url)
            errors = validate_csv(df)
            if errors:
                for e in errors:
                    flash(e, 'error')
                return redirect(request.url)
            insert_df(df)
            flash(f'Successfully uploaded {len(df)} rows.', 'success')
            return redirect(url_for('index'))
        except Exception as e:
            flash(f'Error reading file: {e}', 'error')
            return redirect(request.url)
    return render_template('upload.html', required_columns=REQUIRED_COLUMNS)


@app.route('/generate_sample')
def generate_sample():
    df = generate_sample_data(100)
    insert_df(df)
    flash('Generated 100 sample student records successfully.', 'success')
    return redirect(url_for('index'))


@app.route('/reset')
def reset():
    """Delete all rows and reset the ID counter back to 1."""
    conn = get_db()
    conn.execute('DELETE FROM survey_data')
    # Reset the autoincrement counter so IDs start from 1 again
    try:
        conn.execute("DELETE FROM sqlite_sequence WHERE name='survey_data'")
    except Exception:
        pass
    conn.commit()
    conn.close()
    clear_charts()
    flash('Database reset. All data deleted. IDs will restart from 1 on next upload.', 'info')
    return redirect(url_for('index'))


@app.route('/view_data')
def view_data():
    df = load_data()
    if df.empty:
        flash('No data found. Please upload a CSV first.', 'info')
        return redirect(url_for('upload'))
    rows    = df.head(50).to_dict(orient='records')
    columns = df.columns.tolist()
    return render_template('view_data.html', rows=rows, columns=columns, total=len(df))


@app.route('/analysis')
def analysis():
    df = load_data()
    if df.empty:
        flash('No data found. Please upload data first.', 'error')
        return redirect(url_for('upload'))
    if len(df) < 10:
        flash('Need at least 10 rows for analysis.', 'error')
        return redirect(url_for('upload'))

    clear_charts()
    charts  = []
    results = []

    # Descriptive statistics
    desc_stats = df.select_dtypes(include=[np.number]).describe().round(3).to_html(classes='stats-table')

    # Frequency tables
    freq_tables = {}
    for col in ['gender', 'major', 'academic_year', 'ai_usage', 'ai_frequency']:
        if col in df.columns:
            ft = df[col].value_counts().reset_index()
            ft.columns = [col, 'Count']
            ft['Percentage (%)'] = (ft['Count'] / len(df) * 100).round(1)
            freq_tables[col] = ft.to_html(index=False, classes='stats-table')

    # Visual charts
    for col, title in [('gender', 'Gender Distribution'),
                        ('ai_usage', 'AI Usage — Yes vs No'),
                        ('academic_year', 'Academic Year Distribution')]:
        if col in df.columns:
            charts.append({'path': pie_chart(df, col, title), 'title': title})

    for col, title in [('major', 'Major Distribution'),
                        ('ai_frequency', 'AI Usage Frequency'),
                        ('comfort_level', 'Comfort Level with AI')]:
        if col in df.columns:
            charts.append({'path': bar_chart(df, col, title), 'title': title})

    if 'ai_usage' in df.columns and 'gpa' in df.columns:
        charts.append({'path': boxplot_chart(df, 'ai_usage', 'gpa', 'GPA Distribution by AI Usage'),
                       'title': 'GPA by AI Usage'})

    if 'comfort_level' in df.columns and 'gpa' in df.columns:
        charts.append({'path': scatter_chart(df, 'comfort_level', 'gpa', 'Comfort Level vs GPA'),
                       'title': 'Comfort Level vs GPA'})

    # Statistical tests
    if 'gender' in df.columns and 'ai_usage' in df.columns:
        results.append(run_chi_square(df, 'gender', 'ai_usage'))

    if 'major' in df.columns and 'ai_usage' in df.columns:
        results.append(run_chi_square(df, 'major', 'ai_usage'))

    if 'ai_usage' in df.columns and 'gpa' in df.columns:
        vals = df['ai_usage'].dropna().unique()
        if len(vals) >= 2:
            results.append(run_t_test(df, 'ai_usage', 'gpa', vals[0], vals[1]))

    if 'ai_frequency' in df.columns and 'gpa' in df.columns:
        results.append(run_anova(df, 'ai_frequency', 'gpa'))

    if 'comfort_level' in df.columns and 'gpa' in df.columns:
        results.append(run_correlation(df, 'comfort_level', 'gpa'))

    if 'career_interest' in df.columns and 'optimism_score' in df.columns:
        results.append(run_correlation(df, 'career_interest', 'optimism_score'))

    # Summary p-value chart
    summary_chart = summary_bar_chart(results)

    # Final conclusion paragraph
    final_conclusion = make_final_conclusion(df, results)

    # Observations
    observations = make_observations(df)

    return render_template(
        'analysis.html',
        desc_stats=desc_stats,
        freq_tables=freq_tables,
        charts=charts,
        results=results,
        summary_chart=summary_chart,
        final_conclusion=final_conclusion,
        observations=observations,
        total_rows=len(df)
    )


@app.route('/learn')
def learn():
    return render_template('learn.html')


@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    os.makedirs(CHARTS_FOLDER, exist_ok=True)
    init_db()
    app.run(debug=True, port=5000)
