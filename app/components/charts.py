import plotly.express as px


def bar_chart(df, x, y):
    fig = px.bar(df, x=x, y=y, text=y)
    fig.update_layout(template="plotly_dark")
    return fig


def pie_chart(df, names, values):
    fig = px.pie(df, names=names, values=values, hole=0.4)
    fig.update_layout(template="plotly_dark")
    return fig


def line_chart(df, x, y):
    fig = px.line(df, x=x, y=y)
    fig.update_layout(template="plotly_dark")
    return fig