"""
NYC Housing x Traffic Safety - Dash website (Milestone 1)

Data files (produced by DEm1_revised.ipynb):
  merged_zip.csv         one row per ZIP: borough, centroid, totals (used for ZIP attributes)
  housing_zip_type.csv   affordable housing per ZIP x construction type  -> housing filter
  crashes_zip_year.csv   serious-injury crashes per ZIP x year           -> transport filter

Run locally:  python app.py         Deploy (Render):  gunicorn app:server
"""
import re

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State
from scipy import stats

# ---------------------------------------------------------------
# Constants
# ---------------------------------------------------------------
BOROUGHS = ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island']
BOROUGH_COLORS = {'Bronx': '#1f77b4', 'Brooklyn': '#d62728', 'Manhattan': '#2ca02c',
                  'Queens': '#ff7f0e', 'Staten Island': '#9467bd'}
CONSTRUCTION_TYPES = ['New Construction', 'Preservation']
METRICS = {'crashes': 'Serious-injury crashes', 'fatalities': 'Fatalities',
           'total_injured': 'People injured'}
MIN_UNITS_FOR_RATE = 100                      # same domain rule as the notebook
YEAR_FRACTION = {2022: 1.0, 2023: 1.0, 2024: 1.0, 2025: 1.0, 2026: 181 / 365}  # 2026 = Jan-Jun
TEMPLATE = 'plotly_white'

# ---------------------------------------------------------------
# Load data
# ---------------------------------------------------------------
zip_info = pd.read_csv('merged_zip.csv', dtype={'postcode': str})
housing_type = pd.read_csv('housing_zip_type.csv', dtype={'postcode': str})
crashes_year = pd.read_csv('crashes_zip_year.csv', dtype={'postcode': str})

# A ZIP's dominant construction type is a fixed attribute (computed on all housing,
# so it does not change when the construction-type filter is applied)
_nc = (housing_type.pivot_table(index='postcode', columns='construction_type',
                                values='counted_units', aggfunc='sum').fillna(0))
_dominant = pd.Series(np.where(_nc.get('New Construction', 0) >= _nc.sum(axis=1) / 2,
                               'Mostly new construction', 'Mostly preservation'), index=_nc.index)
zip_info['dominant_type'] = zip_info['postcode'].map(_dominant)   # NaN for ZIPs without housing
zip_info = zip_info[['postcode', 'borough', 'lat', 'lon', 'dominant_type']]
YEARS = sorted(crashes_year['year'].unique().tolist())

print(f'Loaded {len(zip_info)} ZIPs, {len(housing_type)} housing rows, {len(crashes_year)} crash rows')


# ---------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------
def build_zip_table(borough, ctype, year, zips):
    """Re-aggregate housing and crashes for the current filters and join them by ZIP."""
    h = housing_type if ctype == 'All' else housing_type[housing_type['construction_type'] == ctype]
    c = crashes_year if year == 'All' else crashes_year[crashes_year['year'] == int(year)]

    h = h.groupby('postcode')[['buildings', 'counted_units', 'extremely_low_income_units']].sum()
    c = c.groupby('postcode')[['crashes', 'fatalities', 'total_injured']].sum()

    d = zip_info.join(h, on='postcode').join(c, on='postcode')
    count_cols = ['buildings', 'counted_units', 'extremely_low_income_units',
                  'crashes', 'fatalities', 'total_injured']
    d[count_cols] = d[count_cols].fillna(0)

    if borough != 'All':
        d = d[d['borough'] == borough]
    if zips:
        d = d[d['postcode'].isin(zips)]
    # keep ZIPs that have something to show under the current filters
    d = d[(d['counted_units'] > 0) | (d['crashes'] > 0)].copy()

    d['eli_share'] = np.where(d['counted_units'] > 0,
                              d['extremely_low_income_units'] / d['counted_units'] * 100, np.nan)
    return d.reset_index(drop=True)


def add_rate(d, metric):
    d = d.copy()
    d['rate'] = np.where(d['counted_units'] >= MIN_UNITS_FOR_RATE,
                         d[metric] / d['counted_units'] * 1000, np.nan)
    return d


def year_table(d, year):
    """Crash metrics per ZIP x year for the ZIPs in d (respects the year filter)."""
    c = crashes_year[crashes_year['postcode'].isin(d['postcode'])]
    if year != 'All':
        c = c[c['year'] == int(year)]
    return c.merge(d[['postcode', 'borough', 'counted_units']], on='postcode')


# ---------------------------------------------------------------
# Search parser:  "Brooklyn 2022 new buildings" -> borough, year, type
# ---------------------------------------------------------------
BOROUGH_WORDS = {'staten island': 'Staten Island', 'staten': 'Staten Island', 'si': 'Staten Island',
                 'bronx': 'Bronx', 'brooklyn': 'Brooklyn', 'bk': 'Brooklyn',
                 'manhattan': 'Manhattan', 'queens': 'Queens'}
TYPE_WORDS = {'new': 'New Construction', 'construction': 'New Construction',
              'preservation': 'Preservation', 'preserved': 'Preservation', 'preserve': 'Preservation'}
METRIC_WORDS = {'fatal': 'fatalities', 'fatalities': 'fatalities', 'deaths': 'fatalities',
                'killed': 'fatalities', 'injured': 'total_injured', 'injuries': 'total_injured',
                'injury': 'total_injured', 'crash': 'crashes', 'crashes': 'crashes'}
FILLER = {'buildings', 'building', 'units', 'housing', 'affordable', 'in', 'and', 'the', 'of',
          'for', 'zip', 'zips', 'show', 'me', 'all', 'nyc'}


def parse_search(text):
    """Return (filters found, list of words that were not understood)."""
    found, unknown = {}, []
    q = (text or '').lower().strip()
    for phrase, boro in BOROUGH_WORDS.items():              # multi-word names first
        if ' ' in phrase and phrase in q:
            found['borough'] = boro
            q = q.replace(phrase, ' ')
    zips = re.findall(r'\b1[01]\d{3}\b', q)
    if zips:
        found['zips'] = zips
        q = re.sub(r'\b1[01]\d{3}\b', ' ', q)
    for tok in re.findall(r'[a-z0-9]+', q):
        if tok in BOROUGH_WORDS:
            found['borough'] = BOROUGH_WORDS[tok]
        elif re.fullmatch(r'20\d\d', tok):
            if int(tok) in YEARS:
                found['year'] = int(tok)
            else:
                unknown.append(f'{tok} (data covers {YEARS[0]}-{YEARS[-1]})')
        elif tok in TYPE_WORDS:
            found['ctype'] = TYPE_WORDS[tok]
        elif tok in METRIC_WORDS:
            found['metric'] = METRIC_WORDS[tok]
        elif tok not in FILLER:
            unknown.append(tok)
    return found, unknown


# ---------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------
def placeholder(msg):
    fig = px.scatter(title=msg)
    fig.update_layout(template=TEMPLATE, xaxis_visible=False, yaxis_visible=False)
    return fig


def styled(fig, height=430):
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=40, r=20, t=60, b=40),
                      title_font_size=15)
    return fig


def safe(builder, name):
    """Build one chart; a failure in one chart must not break the whole report."""
    try:
        return builder()
    except Exception as err:                           # noqa: BLE001 - shown to the user instead
        print(f'{name} error:', err)
        return placeholder(f'{name} - not enough data for the current filters')


def spearman_text(x, y):
    ok = x.notna() & y.notna()
    if ok.sum() < 5 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return 'rho n/a (fewer than 5 ZIPs)', None
    rho, p = stats.spearmanr(x[ok], y[ok])
    return f'Spearman rho = {rho:.2f} (p = {p:.3f}, n = {ok.sum()})', rho


# ---------------------------------------------------------------
# Layout
# ---------------------------------------------------------------
app = Dash(__name__)
app.title = 'NYC Housing x Traffic Safety'
server = app.server

CARD = {'backgroundColor': 'white', 'borderRadius': '8px', 'padding': '8px',
        'boxShadow': '0 1px 4px rgba(0,0,0,0.12)'}
CAPTION = {'fontSize': '13px', 'color': '#444', 'margin': '4px 8px 8px'}


def chart_card(chart_id, question, height='440px'):
    return html.Div(style=CARD, children=[
        html.P(question, style={**CAPTION, 'fontWeight': 'bold'}),
        dcc.Graph(id=chart_id, style={'height': height}),
    ])


def grid(*children):
    return html.Div(children=list(children),
                    style={'display': 'grid', 'gridTemplateColumns': 'repeat(auto-fit, minmax(520px, 1fr))',
                           'gap': '16px', 'marginBottom': '16px'})


def dropdown(label, id_, options, value):
    return html.Div(style={'flex': '1', 'minWidth': '180px'}, children=[
        html.Label(label, style={'fontWeight': 'bold', 'fontSize': '14px'}),
        dcc.Dropdown(id=id_, options=options, value=value, clearable=False)])


app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f4f5f7', 'padding': '20px'},
    children=html.Div(style={'maxWidth': '1400px', 'margin': 'auto'}, children=[
        html.H1('NYC Housing × Traffic Safety', style={'textAlign': 'center', 'marginBottom': '4px'}),
        html.P('Do the neighbourhoods with the most affordable housing also experience the most '
               'serious traffic crashes? Affordable Housing Production by Building (HPD) × '
               'SIRTA serious-injury crashes 2022 – Jun 2026, joined by ZIP code.',
               style={'textAlign': 'center', 'color': '#555', 'marginTop': 0}),

        # ---- Controls ----
        html.Div(style={**CARD, 'padding': '16px', 'marginBottom': '16px'}, children=[
            html.Div(style={'display': 'flex', 'gap': '14px', 'flexWrap': 'wrap'}, children=[
                dropdown('Borough', 'borough-dd',
                         [{'label': 'All boroughs', 'value': 'All'}] +
                         [{'label': b, 'value': b} for b in BOROUGHS], 'All'),
                dropdown('Housing type (HPD)', 'ctype-dd',
                         [{'label': 'All affordable housing', 'value': 'All'}] +
                         [{'label': t, 'value': t} for t in CONSTRUCTION_TYPES], 'All'),
                dropdown('Crash year (SIRTA)', 'year-dd',
                         [{'label': 'All years (2022 – Jun 2026)', 'value': 'All'}] +
                         [{'label': f'{y}{" (Jan–Jun)" if y == 2026 else ""}', 'value': y}
                          for y in YEARS], 'All'),
                dropdown('Safety metric', 'metric-dd',
                         [{'label': v, 'value': k} for k, v in METRICS.items()], 'crashes'),
            ]),
            html.Div(style={'display': 'flex', 'gap': '12px', 'marginTop': '14px',
                            'alignItems': 'flex-end', 'flexWrap': 'wrap'}, children=[
                html.Div(style={'flex': '3', 'minWidth': '280px'}, children=[
                    html.Label('Search (sets the filters automatically)',
                               style={'fontWeight': 'bold', 'fontSize': '14px'}),
                    dcc.Input(id='search-input', type='text', debounce=False,
                              placeholder='e.g. "Brooklyn 2022 new buildings", "Queens fatalities", '
                                          '"11207 11212"',
                              style={'width': '100%', 'padding': '9px', 'boxSizing': 'border-box'}),
                ]),
                html.Button('Generate Report', id='generate-btn', n_clicks=0,
                            style={'padding': '10px 26px', 'fontSize': '16px', 'cursor': 'pointer',
                                   'backgroundColor': '#0b5cad', 'color': 'white', 'border': 'none',
                                   'borderRadius': '6px', 'height': '40px'}),
            ]),
            html.Div(id='status', style={'marginTop': '10px', 'fontSize': '14px', 'color': '#333'}),
        ]),

        # ---- Charts ----
        dcc.Loading(type='default', children=[
            html.H2('1. Is there a relationship?', style={'fontSize': '20px'}),
            grid(chart_card('chart1', 'RQ1 · Do ZIPs with more affordable units have more serious crashes?'),
                 chart_card('chart2', 'RQ2 · Which borough carries the heaviest burden per affordable unit?')),
            grid(chart_card('chart3', 'RQ3 · Are the highest-crash ZIPs also major housing locations?'),
                 chart_card('chart4', 'RQ4 · Do the biggest housing ZIPs see above-median crashes?')),
            html.H2('2. Where and when?', style={'fontSize': '20px'}),
            chart_card('chart6', 'RQ7 · Do crash hotspots and housing concentrations overlap in space?',
                       height='600px'),
            html.Div(style={'height': '16px'}),
            grid(chart_card('chart5', 'RQ5 · Has the crash burden per unit changed over time, by borough?'),
                 chart_card('chart7', 'RQ8 · Do high-, low- and no-housing ZIPs follow different trends?')),
            html.H2('3. For whom, and what kind of housing?', style={'fontSize': '20px'}),
            grid(chart_card('chart8', 'RQ6 · Is each borough\'s share of crashes proportional to its share of units?'),
                 chart_card('chart9', 'RQ9 · Do ZIPs serving the lowest incomes have higher crash rates?')),
            grid(chart_card('chart10', 'RQ10 · New construction vs. preservation - different crash rates?')),
        ]),

        # ---- Story ----
        html.Div(id='story-section', style={**CARD, 'padding': '22px', 'marginTop': '10px'}),
    ]))


# ---------------------------------------------------------------
# Callback: Generate Report -> parse search, update filters, rebuild all charts + story
# ---------------------------------------------------------------
@app.callback(
    [Output(f'chart{i}', 'figure') for i in range(1, 11)] +
    [Output('story-section', 'children'), Output('status', 'children'),
     Output('borough-dd', 'value'), Output('ctype-dd', 'value'),
     Output('year-dd', 'value'), Output('metric-dd', 'value')],
    Input('generate-btn', 'n_clicks'),
    State('borough-dd', 'value'), State('ctype-dd', 'value'),
    State('year-dd', 'value'), State('metric-dd', 'value'),
    State('search-input', 'value'),
)
def generate_report(n_clicks, borough, ctype, year, metric, search):
    # 1) Search terms override the dropdowns (and the dropdowns are updated to match)
    found, unknown = parse_search(search)
    borough = found.get('borough', borough)
    ctype = found.get('ctype', ctype)
    year = found.get('year', year)
    metric = found.get('metric', metric)
    zips = found.get('zips', [])
    mlabel = METRICS[metric]

    status_parts = [f'Borough: {borough}', f'Housing type: {ctype}', f'Crash year: {year}',
                    f'Metric: {mlabel}']
    if zips:
        status_parts.append('ZIPs: ' + ', '.join(zips))
    status = [html.B('Active filters - '), ' · '.join(status_parts)]
    if unknown:
        status += [html.Span(f'   (ignored search words: {", ".join(unknown)})',
                             style={'color': '#b00'})]

    # 2) Data for the current selection
    d = add_rate(build_zip_table(borough, ctype, year, zips), metric)
    if d.empty:
        empty = placeholder('No data matches the current filters')
        story = html.Div([html.H3('No results'),
                          html.P('Try a different borough, year, housing type or search.')])
        return [empty] * 10 + [story, status, borough, ctype, year, metric]

    hz = d[d['counted_units'] > 0]
    rated = d[d['rate'].notna()]
    rate_label = f'{mlabel} per 1,000 affordable units'
    period = 'Jan–Jun 2026' if year == 2026 else (str(year) if year != 'All' else '2022 – Jun 2026')
    rho_txt, rho = spearman_text(hz['counted_units'], hz[metric])

    # ---- Chart 1: scatter units vs metric ----
    def c1():
        if len(hz) == 0:
            return placeholder('Chart 1 - no ZIPs with affordable housing in this selection')
        fig = px.scatter(hz, x='counted_units', y=metric, color='borough', log_x=True,
                         size=np.sqrt(hz['fatalities'] + 1), size_max=16, hover_name='postcode',
                         color_discrete_map=BOROUGH_COLORS, category_orders={'borough': BOROUGHS},
                         hover_data={'counted_units': ':,', metric: True, 'fatalities': True},
                         title=f'Chart 1 - Affordable units vs. {mlabel.lower()} per ZIP<br>'
                               f'<sup>{rho_txt} · bubble size = fatalities</sup>',
                         labels={'counted_units': 'Affordable units in ZIP (log scale)',
                                 metric: f'{mlabel} ({period})', 'borough': 'Borough',
                                 'fatalities': 'Fatalities', 'size': 'size'})
        return styled(fig)

    # ---- Chart 2: borough rate ----
    def c2():
        b = d.groupby('borough')[['counted_units', metric]].sum().reset_index()
        b = b[b['counted_units'] > 0]
        b['rate'] = b[metric] / b['counted_units'] * 1000
        fig = px.bar(b, x='borough', y='rate', color='borough', color_discrete_map=BOROUGH_COLORS,
                     text_auto='.1f', category_orders={'borough': BOROUGHS},
                     hover_data={'counted_units': ':,', metric: ':,'},
                     title=f'Chart 2 - {rate_label}, by borough',
                     labels={'rate': rate_label, 'borough': 'Borough',
                             'counted_units': 'Affordable units', metric: mlabel})
        fig.add_hline(y=b[metric].sum() / b['counted_units'].sum() * 1000, line_dash='dash',
                      annotation_text='Selection average')
        fig.update_layout(showlegend=False)
        return styled(fig)

    # ---- Chart 3: top ZIPs by metric, coloured by housing percentile ----
    def c3():
        t = d.assign(units_pct=d['counted_units'].rank(pct=True) * 100)
        t = t.nlargest(min(10, len(t)), metric).sort_values(metric)
        fig = px.bar(t, x=metric, y='postcode', orientation='h', color='units_pct',
                     color_continuous_scale='Viridis', range_color=[0, 100],
                     text=t['units_pct'].round(0).astype(int).astype(str) + 'th pct',
                     hover_data={'borough': True, 'counted_units': ':,'},
                     title=f'Chart 3 - Top {len(t)} ZIPs by {mlabel.lower()}, coloured by housing percentile',
                     labels={metric: mlabel, 'postcode': 'ZIP code', 'units_pct': 'Housing pct',
                             'counted_units': 'Affordable units', 'borough': 'Borough'})
        fig.update_layout(yaxis_type='category')
        return styled(fig)

    # ---- Chart 4: top housing ZIPs vs median metric ----
    def c4():
        if len(hz) == 0:
            return placeholder('Chart 4 - no ZIPs with affordable housing in this selection')
        med = d[metric].median()
        t = hz.nlargest(min(10, len(hz)), 'counted_units').sort_values('counted_units')
        t = t.assign(vs=np.where(t[metric] > med, 'Above median', 'At/below median'))
        fig = px.bar(t, x=metric, y='postcode', orientation='h', color='vs', text=metric,
                     color_discrete_map={'Above median': '#d62728', 'At/below median': '#7f7f7f'},
                     hover_data={'borough': True, 'counted_units': ':,'},
                     title=f'Chart 4 - {mlabel} in the {len(t)} ZIPs with the most affordable units',
                     labels={metric: mlabel, 'postcode': 'ZIP (ordered by affordable units)', 'vs': '',
                             'counted_units': 'Affordable units', 'borough': 'Borough'})
        fig.add_vline(x=med, line_dash='dash', annotation_text=f'Median ZIP: {med:.0f}')
        fig.update_layout(yaxis_type='category')
        return styled(fig)

    # ---- Chart 5: heatmap borough x year, rate per 1,000 units (annualised) ----
    def c5():
        y = year_table(d, year)
        units = d.groupby('borough')['counted_units'].sum()
        g = y.groupby(['borough', 'year'])[metric].sum().reset_index()
        g['rate'] = g[metric] / g['year'].map(YEAR_FRACTION) / g['borough'].map(units) * 1000
        heat = g.pivot(index='borough', columns='year', values='rate')
        heat = heat.reindex([b for b in BOROUGHS if b in heat.index and units.get(b, 0) > 0])
        heat.columns = [f'{c}*' if c == 2026 else str(c) for c in heat.columns]
        fig = px.imshow(heat, text_auto='.1f', color_continuous_scale='Reds', aspect='auto',
                        title=f'Chart 5 - {rate_label} per year (*2026 annualised)',
                        labels={'x': 'Crash year', 'y': 'Borough', 'color': 'Per 1,000 units'})
        return styled(fig)

    # ---- Chart 6: map ----
    def c6():
        m = d.dropna(subset=['lat', 'lon']).copy()
        m['units_colour'] = m['counted_units'].clip(upper=max(m['counted_units'].quantile(0.95), 1))
        m['bubble'] = m[metric] + 0.5            # keep zero-crash ZIPs visible
        fig = px.scatter_map(
            m, lat='lat', lon='lon', size='bubble', size_max=22, color='units_colour',
            color_continuous_scale='Viridis', hover_name='postcode',
            hover_data={'borough': True, 'counted_units': ':,', metric: True,
                        'lat': False, 'lon': False, 'units_colour': False, 'bubble': False},
            zoom=9.3, center={'lat': 40.71, 'lon': -73.93}, map_style='carto-positron',
            title=f'Chart 6 - Bubble size = {mlabel.lower()}, colour = affordable units (capped at 95th pct)',
            labels={'units_colour': 'Affordable units', 'counted_units': 'Affordable units',
                    metric: mlabel, 'borough': 'Borough'})
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0), height=590, title_font_size=15)
        return fig

    # ---- Chart 7: trend by housing group ----
    def c7():
        g = d.copy()
        g['group'] = 'No affordable housing'
        has = g['counted_units'] > 0
        if has.sum() >= 3:
            g.loc[has, 'group'] = pd.qcut(g.loc[has, 'counted_units'].rank(method='first'), 3,
                                          labels=['Low housing', 'Medium housing', 'High housing']).astype(str)
        else:
            g.loc[has, 'group'] = 'With housing'
        y = year_table(g, year).merge(g[['postcode', 'group']], on='postcode')
        t = y.groupby(['group', 'year'])[metric].sum().reset_index()
        t['per_zip'] = t[metric] / t['year'].map(YEAR_FRACTION) / t['group'].map(g['group'].value_counts())
        fig = px.line(t, x='year', y='per_zip', color='group', markers=True,
                      category_orders={'group': ['No affordable housing', 'Low housing',
                                                 'Medium housing', 'High housing', 'With housing']},
                      title=f'Chart 7 - {mlabel} per ZIP per year, by housing group (2026 annualised)',
                      labels={'year': 'Year', 'per_zip': f'{mlabel} per ZIP per year',
                              'group': 'ZIP group (terciles of units)'})
        fig.update_layout(xaxis=dict(dtick=1))
        return styled(fig)

    # ---- Chart 8: share of units vs share of metric ----
    def c8():
        b = d.groupby('borough')[['counted_units', metric]].sum()
        if b['counted_units'].sum() == 0 or b[metric].sum() == 0:
            return placeholder('Chart 8 - nothing to compare in this selection')
        s = pd.DataFrame({'Share of affordable units': b['counted_units'] / b['counted_units'].sum() * 100,
                          f'Share of {mlabel.lower()}': b[metric] / b[metric].sum() * 100})
        s = s.reindex([x for x in BOROUGHS if x in s.index]).reset_index().melt(
            id_vars='borough', var_name='measure', value_name='share')
        fig = px.bar(s, x='borough', y='share', color='measure', barmode='group', text_auto='.1f',
                     title=f'Chart 8 - Share of affordable units vs. share of {mlabel.lower()}',
                     labels={'share': 'Share of selection total (%)', 'borough': 'Borough', 'measure': ''})
        fig.update_layout(legend=dict(orientation='h', y=-0.2))
        return styled(fig)

    # ---- Chart 9: ELI share vs rate ----
    def c9():
        r = rated[rated['rate'] > 0]
        if len(r) < 3:
            return placeholder(f'Chart 9 - fewer than 3 ZIPs with >= {MIN_UNITS_FOR_RATE} units')
        txt, _ = spearman_text(r['eli_share'], r['rate'])
        fig = px.scatter(r, x='eli_share', y='rate', color='borough', size='counted_units', size_max=24,
                         log_y=True, hover_name='postcode', color_discrete_map=BOROUGH_COLORS,
                         category_orders={'borough': BOROUGHS},
                         title=f'Chart 9 - Extremely-low-income share vs. rate<br><sup>{txt}</sup>',
                         labels={'eli_share': 'Extremely-low-income units (% of affordable units)',
                                 'rate': f'{rate_label} (log)', 'borough': 'Borough',
                                 'counted_units': 'Affordable units'})
        return styled(fig)

    # ---- Chart 10: dominant construction type ----
    def c10():
        r = rated[rated['rate'] > 0]
        if r['dominant_type'].nunique() < 2:
            return placeholder('Chart 10 - needs ZIPs of both housing types')
        a = r.loc[r['dominant_type'] == 'Mostly new construction', 'rate']
        b = r.loc[r['dominant_type'] == 'Mostly preservation', 'rate']
        p = stats.mannwhitneyu(a, b)[1] if min(len(a), len(b)) >= 3 else np.nan
        fig = px.box(r, x='dominant_type', y='rate', color='dominant_type', points='all', log_y=True,
                     hover_name='postcode', hover_data={'borough': True, 'counted_units': ':,'},
                     category_orders={'dominant_type': ['Mostly new construction', 'Mostly preservation']},
                     title=f'Chart 10 - {rate_label} by dominant housing type'
                           f'<br><sup>Mann-Whitney p = {p:.3f} · compare within one borough before concluding</sup>',
                     labels={'dominant_type': 'Majority of the ZIP\'s affordable units are...',
                             'rate': f'{rate_label} (log)', 'borough': 'Borough',
                             'counted_units': 'Affordable units'})
        fig.update_layout(showlegend=False)
        return styled(fig)

    figs = [safe(f, f'Chart {i}') for i, f in enumerate([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10], 1)]

    # ---- Story (live numbers for the current selection + fixed findings from the notebook) ----
    top = d.nlargest(1, metric).iloc[0]
    top_pct = d['counted_units'].rank(pct=True)[d[metric].idxmax()] * 100
    bt = d.groupby('borough')[['counted_units', metric]].sum()
    bt = bt[bt['counted_units'] >= 1000]
    bt['rate'] = bt[metric] / bt['counted_units'] * 1000
    live = [
        html.Li(f'{len(d)} ZIP codes · {int(d["counted_units"].sum()):,} affordable units · '
                f'{int(d[metric].sum()):,} {mlabel.lower()} ({period}).'),
        html.Li(f'Relationship between affordable units and {mlabel.lower()} across ZIPs: {rho_txt}' +
                ('' if rho is None else
                 (' - a clear positive link.' if rho >= 0.4 else
                  ' - a weak link.' if rho >= 0.2 else ' - no meaningful link.'))),
        html.Li(f'Highest {mlabel.lower()}: ZIP {top["postcode"]} ({top["borough"]}) with '
                f'{int(top[metric]):,}; it ranks in the {top_pct:.0f}th percentile for affordable housing.'),
    ]
    if len(bt) >= 2:
        hi, lo = bt['rate'].idxmax(), bt['rate'].idxmin()
        live.append(html.Li(f'Highest burden per 1,000 affordable units: {hi} ({bt.loc[hi, "rate"]:.1f}); '
                            f'lowest: {lo} ({bt.loc[lo, "rate"]:.1f}).'))

    story = html.Div([
        html.H2('The story', style={'marginTop': 0}),
        html.P([html.B('Question. '), 'Do the NYC neighbourhoods receiving the most affordable housing also '
                'experience the most serious traffic crashes? If they do, the residents with the fewest '
                'alternatives to walking and transit are the most exposed.']),
        html.P([html.B('Data. '), 'HPD Affordable Housing Production by Building (hg8x-zxpr; 7,470 buildings '
                'after removing confidential and duplicate rows) and SIRTA serious-injury crashes '
                '(xeqp-qz8h; Jan 2022 – Jun 2026). Both were aggregated to ZIP code and combined with a '
                'full outer join; missing crash ZIPs were recovered from coordinates (94 % accurate in '
                'validation).']),
        html.H3('Your current selection'),
        html.Ul(live),
        html.H3('Key findings (all ZIPs, all years)'),
        html.Ul([
            html.Li('Affordable housing and serious crashes co-locate: Spearman rho = 0.52 across 158 ZIPs. '
                    'Half of the 20 highest-crash ZIPs are also top-20 housing ZIPs (south Bronx, '
                    'East New York, Brownsville).'),
            html.Li('The burden per unit is uneven: Staten Island (71) and Queens (66) have about 3x the '
                    'Bronx\'s 22 serious crashes per 1,000 affordable units, and Queens\' rate rose 39 % '
                    'from 2022 to 2025.'),
            html.Li('Large edge-of-city complexes (Co-op City 10475, Starrett City 11239, Far Rockaway 11691) '
                    'have very high housing but below-median crashes.'),
            html.Li('Income mix does not predict crash rate (rho = -0.05), and the apparent new-construction '
                    'effect disappears within each borough.'),
        ]),
        html.H3('Limitations'),
        html.Ul([
            html.Li('Correlation, not causation: population, traffic volume and street design are not in '
                    'the data and likely drive both housing density and crashes.'),
            html.Li('Rates are per affordable unit, not per resident; a crash in a ZIP may not involve '
                    'its residents.'),
            html.Li('14 % of crashes could not be placed in a ZIP; housing data starts in 2014 but crash '
                    'data only in 2022; 2026 covers six months.'),
        ]),
        html.H3('Why it matters for NYC'),
        html.Ul([
            html.Li('DOT / Vision Zero: prioritise ZIPs that are both top-housing and top-crash '
                    '(10451–10457, 11207, 11208, 11212).'),
            html.Li('HPD / City Planning: pair new affordable housing in Queens and southeast Brooklyn '
                    'with street-safety review, since these areas carry the highest crash burden per unit.'),
            html.Li('Advocates and community boards: Charts 2 and 8 show Queens is under-served if '
                    'safety money follows housing volume.'),
        ]),
    ])

    return figs + [story, status, borough, ctype, year, metric]


if __name__ == '__main__':
    app.run(debug=True)
