# Typer 
# 轻量级命令行工具

import typer # For building CLI applications, command-line interfaces
from alphafold3_seqvis_toolkit.modules.contact_map_visualization_with_track import contact_map_vis_with_track
from alphafold3_seqvis_toolkit.modules.contact_map_visualization_without_track import contact_map_vis_without_track

@app.command(
    "contact-map-vis-Track",
    no_args_is_help=True
)
def contact_map_vis_with_track_cmd(
    mmcif_file: str = typer.Option(..., "--mmcif-file", help="Path to mmCIF file", rich_help_panel="Input"),
    chains: Optional[List[str]] = typer.Option(None, "--chains", "-c", help="Repeatable: chain IDs to include", rich_help_panel="Input"),
    out_path: str = typer.Option(".", "--out-path", "-o", help="Directory for outputs, default is current directory", rich_help_panel="Output"),
    track_bed_file: Optional[str] = typer.Option(..., "--track-bed-file", help="Path to BED file for custom tracks", rich_help_panel="Custom Tracks"),
    color_config: Optional[str] = typer.Option("tab10", "--color-config", help="Path to color config file (JSON) or colormap name", rich_help_panel="Custom Tracks"),
    tick_step: int = typer.Option(100, "--tick-step", help="Step size for ticks on the axes", rich_help_panel="Custom Tracks"),
):
    """
    Visualize contact map from an AlphaFold3 mmCIF structure or a general mmCIF structure with customizable feature annotation tracks.
    
    \b
    Notes:
    - 1, Designed for visualizing contact maps from AlphaFold3 mmCIF outputs or general mmCIF structures with customizable annotation tracks.
    - 2, By default, all chains in the mmCIF file are included. Use --chains to specify particular chains if needed.
    - 3, Custom annotation tracks can be added using a BED file format. Refer to the documentation for details on the required format.
    And tracks can be numerical (line plot) or categorical (bar/strip plot).
    - 4, A color configuration file can be provided to customize the colors of categorical tracks. Refer to the documentation for the required format.
    - 5, The track bed file must be 0-based indexed! All residue indices in this module are 0-based logic driven.
    - 6, If you do not need custom annotation tracks, please use contact-map-vis-noTrack command instead.
    - 7, Modify the tick_step parameter to adjust the spacing of residue ticks on the axes as needed.

    \b
    Examples:
    - 1, Basic contact map visualization:
    af3-vis contact-map-vis --mmcif-file model.cif -o out_path
    - 2, Contact map with custom annotation tracks (e.g., domains, IDRs):
    af3-vis contact-map-vis \
--mmcif-file model.cif \
--track-bed-file custom_tracks.bed \
--color-config color_config.json \
-o out_path
    
    """

    contact_map_vis_with_track(
        mmcif_file=mmcif_file,
        chains=chains,
        out_path=out_path,
        track_bed_file=track_bed_file,
        color_config=color_config,
        tick_step=tick_step,
    )


@app.command(
    "contact-map-vis-noTrack",
    no_args_is_help=True
)
def contact_map_vis_without_track_cmd(
    mmcif_file: str = typer.Option(..., "--mmcif-file", help="Path to mmCIF file", rich_help_panel="Input"),
    chains: Optional[List[str]] = typer.Option(None, "--chains", "-c", help="Repeatable: chain IDs to include", rich_help_panel="Input"),
    out_path: str = typer.Option(".", "--out-path", "-o", help="Directory for outputs", rich_help_panel="Output"),
    tick_step: int = typer.Option(100, "--tick-step", help="Step size for ticks on the axes", rich_help_panel="Custom Tracks"),
):
    """
    Visualize contact map from an AlphaFold3 mmCIF structure or a general mmCIF structure without feature annotation tracks.

    \b
    Notes:
    - 1, Designed for visualizing contact maps from AlphaFold3 mmCIF outputs or general mmCIF structures without annotation tracks.
    - 2, By default, all chains in the mmCIF file are included. Use --chains to specify particular chains if needed.
    - 3, If you need custom annotation tracks, please use contact-map-vis-Track command instead.
    - 4, Modify the tick_step parameter to adjust the spacing of residue ticks on the axes as needed.

    \b
    Examples:
    - 1, Basic contact map visualization:
    af3-vis contact-map-vis-noTrack --mmcif-file model.cif -o out_path

    """

    contact_map_vis_without_track(
        mmcif_file=mmcif_file,
        chains=chains,
        out_path=out_path,
        tick_step=tick_step,
    )
