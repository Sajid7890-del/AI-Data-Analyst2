import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pandas as pd
import numpy as np

# Premium Color Palette
PRIMARY_COLOR = "#636EFA"
SECONDARY_COLOR = "#EF553B"
ACCENT_COLOR = "#00CC96"
COLOR_SEQUENCE = px.colors.qualitative.Plotly

def get_plotly_theme(theme_mode="Dark"):
    """Get the plotly template name based on the current theme mode."""
    return "plotly_dark" if theme_mode == "Dark" else "plotly_white"

# --- Plotly Interactive Charts ---

def plot_bar(df, x_col, y_col, title, color_col=None, theme_mode="Dark"):
    """Generate an interactive Plotly Bar Chart."""
    fig = px.bar(
        df, x=x_col, y=y_col, color=color_col,
        title=title, template=get_plotly_theme(theme_mode),
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_layout(
        title_x=0.5,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Outfit, Inter, sans-serif")
    )
    return fig

def plot_line(df, x_col, y_col, title, color_col=None, theme_mode="Dark"):
    """Generate an interactive Plotly Line Chart."""
    # Sort by X column if it's a date or numerical sequence to make the line look correct
    df_sorted = df.copy()
    try:
        df_sorted = df_sorted.sort_values(by=x_col)
    except:
        pass
        
    fig = px.line(
        df_sorted, x=x_col, y=y_col, color=color_col,
        title=title, template=get_plotly_theme(theme_mode),
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_layout(
        title_x=0.5,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Outfit, Inter, sans-serif")
    )
    return fig

def plot_pie(df, names_col, values_col, title, theme_mode="Dark"):
    """Generate an interactive Plotly Pie Chart."""
    # Aggregate data if duplicate names exist
    df_grouped = df.groupby(names_col)[values_col].sum().reset_index()
    fig = px.pie(
        df_grouped, names=names_col, values=values_col,
        title=title, template=get_plotly_theme(theme_mode),
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_layout(
        title_x=0.5,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Outfit, Inter, sans-serif")
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_histogram(df, col, title, bins=30, theme_mode="Dark"):
    """Generate an interactive Plotly Histogram."""
    fig = px.histogram(
        df, x=col, nbins=bins,
        title=title, template=get_plotly_theme(theme_mode),
        color_discrete_sequence=[PRIMARY_COLOR]
    )
    fig.update_layout(
        title_x=0.5,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Outfit, Inter, sans-serif"),
        bargap=0.1
    )
    return fig

def plot_scatter(df, x_col, y_col, title, color_col=None, theme_mode="Dark"):
    """Generate an interactive Plotly Scatter Plot."""
    fig = px.scatter(
        df, x=x_col, y=y_col, color=color_col,
        title=title, template=get_plotly_theme(theme_mode),
        color_discrete_sequence=COLOR_SEQUENCE
    )
    fig.update_layout(
        title_x=0.5,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Outfit, Inter, sans-serif")
    )
    return fig

def plot_box(df, y_col, x_col=None, title=None, theme_mode="Dark"):
    """Generate an interactive Plotly Box Plot."""
    fig = px.box(
        df, x=x_col, y=y_col,
        title=title or f"Box Plot of {y_col}", template=get_plotly_theme(theme_mode),
        color_discrete_sequence=[SECONDARY_COLOR]
    )
    fig.update_layout(
        title_x=0.5,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Outfit, Inter, sans-serif")
    )
    return fig

def plot_heatmap(corr_df, title="Correlation Matrix", theme_mode="Dark"):
    """Generate an interactive Plotly Heatmap."""
    if corr_df.empty:
        return None
        
    z = corr_df.values
    x = list(corr_df.columns)
    y = list(corr_df.index)
    
    # Custom annotations
    annotations = []
    for i in range(len(y)):
        for j in range(len(x)):
            annotations.append(
                dict(
                    x=x[j], y=y[i], text=str(z[i][j]),
                    xref='x1', yref='y1',
                    showarrow=False,
                    font=dict(color="white" if abs(z[i][j]) > 0.5 else "black" if theme_mode == "Light" else "white")
                )
            )
            
    fig = go.Figure(
        data=go.Heatmap(
            z=z, x=x, y=y,
            colorscale='RdBu_r', zmin=-1, zmax=1
        )
    )
    fig.update_layout(
        title=title, title_x=0.5,
        template=get_plotly_theme(theme_mode),
        annotations=annotations,
        margin=dict(l=40, r=40, t=50, b=40),
        font=dict(family="Outfit, Inter, sans-serif")
    )
    return fig


# --- Matplotlib Static Charts (For Reports) ---

def generate_static_plot(df, plot_type, x_col=None, y_col=None, title="Chart", color_theme="dark"):
    """Generate a static Matplotlib plot and return its image bytes (PNG)."""
    # Set style
    if color_theme == "dark":
        plt.style.use('dark_background')
        bg_color = "#121212"
        text_color = "#ffffff"
        grid_color = "#333333"
    else:
        plt.style.use('default')
        bg_color = "#ffffff"
        text_color = "#000000"
        grid_color = "#e0e0e0"

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.spines['bottom'].set_color(text_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(text_color)
    ax.tick_params(colors=text_color)
    ax.yaxis.label.set_color(text_color)
    ax.xaxis.label.set_color(text_color)
    ax.title.set_color(text_color)
    ax.grid(True, linestyle='--', alpha=0.5, color=grid_color)
    
    # Custom colors
    color_palette = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
    
    try:
        if plot_type == 'bar':
            # Aggregate or plot direct
            if df[x_col].nunique() > 20:
                # Top 15 categories
                df_top = df.groupby(x_col)[y_col].sum().reset_index().sort_values(by=y_col, ascending=False).head(15)
                ax.bar(df_top[x_col], df_top[y_col], color=color_palette[0])
                plt.xticks(rotation=45, ha='right')
            else:
                df_grouped = df.groupby(x_col)[y_col].sum().reset_index()
                ax.bar(df_grouped[x_col], df_grouped[y_col], color=color_palette[0])
                plt.xticks(rotation=45 if df_grouped[x_col].nunique() > 5 else 0)
            ax.set_ylabel(y_col)
            ax.set_xlabel(x_col)
            
        elif plot_type == 'line':
            df_sorted = df.copy()
            try:
                df_sorted = df_sorted.sort_values(by=x_col)
            except:
                pass
            df_grouped = df_sorted.groupby(x_col)[y_col].mean().reset_index()
            ax.plot(df_grouped[x_col], df_grouped[y_col], color=color_palette[0], marker='o', linewidth=2)
            plt.xticks(rotation=45 if df_grouped[x_col].nunique() > 5 else 0)
            ax.set_ylabel(y_col)
            ax.set_xlabel(x_col)
            
        elif plot_type == 'pie':
            df_grouped = df.groupby(x_col)[y_col].sum().reset_index().sort_values(by=y_col, ascending=False).head(8)
            ax.pie(df_grouped[y_col], labels=df_grouped[x_col], colors=color_palette, autopct='%1.1f%%', 
                   textprops={'color': text_color}, startangle=90)
            ax.axis('equal')
            
        elif plot_type == 'histogram':
            ax.hist(df[x_col].dropna(), bins=20, color=color_palette[1], edgecolor='black', alpha=0.8)
            ax.set_xlabel(x_col)
            ax.set_ylabel("Frequency")
            
        elif plot_type == 'scatter':
            ax.scatter(df[x_col], df[y_col], color=color_palette[2], alpha=0.7, edgecolors='none')
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            
        elif plot_type == 'box':
            # Create a boxplot
            clean_data = [group.dropna().values for name, group in df.groupby(x_col)[y_col]] if x_col else [df[y_col].dropna().values]
            labels = [str(name) for name, _ in df.groupby(x_col)[y_col]] if x_col else [y_col]
            bp = ax.boxplot(clean_data, labels=labels, patch_artist=True)
            for box in bp['boxes']:
                box.set(facecolor=color_palette[3], alpha=0.8)
            for median in bp['medians']:
                median.set(color="#ffffff", linewidth=2)
            ax.set_ylabel(y_col)
            if x_col:
                ax.set_xlabel(x_col)
                
        elif plot_type == 'heatmap':
            # Correlation Matrix heatmap
            corr_df = df
            sns.heatmap(corr_df, annot=True, cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, cbar=True)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
        ax.set_title(title, fontsize=14, pad=15)
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        plt.close(fig)
        return buf.getvalue()
    except Exception as e:
        plt.close(fig)
        print(f"Error creating static plot: {e}")
        return None
