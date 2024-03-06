def plot():
    root_path = '../../'
    file = f'{root_path}/outputs/data_stats.csv'
    from efficiency.log import fread
    df = fread(file, return_df=True)

    import plotly.express as px
    import plotly.io as pio

    # model_version, system_role, lang = file_name.split('_', 2)
    custom_color_scale = ["#FCC9A5", "#EE977F", "#9BBFE0", "#95D0E0", "#7BC3DA", "#61B6D4", "#47A9CE"]
    # ["#f2d9e6", "#e3b5bc", "#d290a1", "#c26b87", "#b3456c", "#a41f52", "#95003a"]
    fig = px.sunburst(df, path=['layer1', 'layer2'],
                      values='percentage', color='layer1',
                      color_discrete_sequence=custom_color_scale)
    fig.update_layout(font_size=30)
    # fig.show()
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=20),  # Set bottom margin large for cropping later, as the bottom left corner
        # has some erraneous words: "Loading [MathJax]/extensions/MathMenu.js"
    )
    plot_file = f"{root_path}/outputs/data_stats.pdf"
    pio.write_image(fig, plot_file, format="pdf", width=600, height=600, scale=2)
    print(f'[Info] Generated figures in {plot_file}')

plot()
