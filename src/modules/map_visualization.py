from geojson.feature import Feature, FeatureCollection
import json
import folium
import branca.colormap as cm
import h3

# These functions are heavily inspired by https://github.com/uber/h3-py-notebooks


def hexagons_dataframe_to_geojson(
    df_hex, hex_id_field, geometry_field, value_field, file_output=None
):
    """Produce the GeoJSON representation containing all geometries in a dataframe
    based on a column in geojson format (geometry_field)"""

    list_features = []

    for i, row in df_hex.iterrows():
        feature = Feature(
            geometry=row[geometry_field],
            id=row[hex_id_field],
            properties={"value": row[value_field]},
        )
        list_features.append(feature)

    feat_collection = FeatureCollection(list_features)

    geojson_result = json.dumps(feat_collection)

    # optionally write to file
    if file_output is not None:
        with open(file_output, "w") as f:
            json.dump(feat_collection, f)

    return geojson_result


def choropleth_map(
    df_aggreg,
    hex_id_field,
    geometry_field,
    value_field,
    layer_name,
    kind="linear",
    border_color="black",
    fill_opacity=0.7,
    with_legend=False,
):

    """Plots a choropleth map with folium"""

    plot_map = folium.Map(
        location=(50.94, 6.94),
        zoom_start=9.5,
    )

    # the custom colormap depends on the map kind
    if kind == "linear":
        min_value = df_aggreg[value_field].min()
        max_value = df_aggreg[value_field].max()
        m = round((min_value + max_value) / 2, 0)
        custom_cm = cm.LinearColormap(
            ["green", "yellow", "red"], vmin=min_value, vmax=max_value
        )
    elif kind == "outlier":
        # for outliers, values would be -1,0,1
        custom_cm = cm.LinearColormap(["blue", "white", "red"], vmin=-1, vmax=1)
    elif kind == "filled_nulls":
        min_value = df_aggreg[df_aggreg[value_field] > 0][value_field].min()
        max_value = df_aggreg[df_aggreg[value_field] > 0][value_field].max()
        m = round((min_value + max_value) / 2, 0)
        custom_cm = cm.LinearColormap(
            ["silver", "green", "yellow", "red"],
            index=[0, min_value, m, max_value],
            vmin=min_value,
            vmax=max_value,
        )

    # create geojson data from dataframe
    geojson_data = hexagons_dataframe_to_geojson(
        df_aggreg, hex_id_field, geometry_field, value_field
    )

    # plot on map
    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            "fillColor": custom_cm(feature["properties"]["value"]),
            "color": border_color,
            "weight": 1,
            "fillOpacity": fill_opacity,
        },
        # name = layer_name
    ).add_to(plot_map)

    # add legend (not recommended if multiple layers)
    if with_legend is True:
        custom_cm.add_to(plot_map)

    return plot_map

    def base_empty_map():
        return m