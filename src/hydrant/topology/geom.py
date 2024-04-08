"""
Common functions for working with river and sub-basin geometries
in the context of hydrological modelling
"""

import geopandas as gpd
import pandas as pd

from typing import (
    Optional,
    Dict,
    Tuple,
    Union,
    Sequence,
)

from .river_graph import find_upstream


def prepare_cat(
    cat: gpd.GeoDataFrame,
    cat_col_id: str,
    cst: Optional[gpd.GeoDataFrame] = None,
    cst_col_mapper: Optional[Dict[str, str]] = None,
    cst_col_id: Optional[str] = None,
    cst_colid_reset: bool = True,
    crs: int = 4326,
    *args,
) -> gpd.GeoDataFrame:
    '''Preparing `catchment` object as part of the mizuroute setup workflow
    
    Parameters
    ----------
    cat: str or geopandas.GeoDataFrame
        The address or the geopandas.GeoDataFrame object of the `catchment`
        layer
    cat_col_id: str
        The column name of `cat` indicating IDs of available geometries
    nca: str or geopandas.GeoDataFrame, optional [defaults to None]
        The address or the geopandas.GeoDataFrame object of the
        non-contributing catchment layer
    nca_col_map: dict, optional [defaults to None]
        Dictionary used for renaming necessary columns of `nca`
    nca_col_id: str, optional [defaults to None]
        The column name of `nca` indicating IDs of available geometries
    nca_col_id_reset: bool, optional [defaults to True]
        The flag to re-index the non-contributing catchments; if
        re-indexed the numbers will be assigned to continue that of
        `cat`'s
    crs: int, optional [defaults to 4326]
        The Coordination Reference System (CRS) EPSG (integer) number
    *args: iterable
        List of all desired columns to be included in the returned
        geopandas.GeoDataFrame object
    
    Returns
    -------
    catchment: geopandas.GeoDataFrame
        The catchment layer including columns for the geometry IDs, 
    '''
    # define necessary variables
    area_id = 'unitarea'
    geometry_id = 'geometry'
    hillslope_id = 'hillslope'
    
    # create of copy of the input GeoDataFrame variables
    cst = cst.copy()
    cat = cat.copy()
    
    # `cat` must be of type `geopandas.GeoDataFrame`
    assert isinstance(cat, gpd.GeoDataFrame), "`cat` must be of type `geopandas.GeoDataFrame`"
    # `cat_col_id` must be of type `str`
    assert isinstance(cat_col_id, str), "`cat_col_id` must be of type `geopandas.GeoDataFrame`"
    
    if cst_col_mapper is not None:
        assert isinstance(cst_col_mapper, dict), "`cst_col_mapper` must be of type `dict`"
        
    # assign a flag for `cat`
    cat[hillslope_id] = 0
    
    if cst is not None:
        # assign CRS
        if not cst.crs:
            cst.set_crs(epsg=4326, inplace=True)
            warnings.warn('CRS of the coastal hillslope Shapefile has been assumed to be EPSG:4326')
        # change column names if necessary
        if cst_col_mapper is not None:
            cst.rename(columns=cst_col_mapper, inplace=True)
        
        # reset index
        cst.reset_index(drop=True, inplace=True)
        
        # assigning new ID values
        if cst_colid_reset:
            cst[cat_col_id] = range(cat[cat_col_id].max()+1,
                                    cat[cat_col_id].max()+1+len(cst))
        
        # calculating unit area in km2
        # first transforming crs to an equal area one, i.e., EPSG:6933
        cst[area_id] = cst.to_crs(epsg=6933).area / 1e6
        
        # re-arraning columns
        cst.reindex(columns=[cat_col_id, area_id, geometry_id, *args])
        
        # assign `coastal_hillslope` flag
        cst[hillslope_id] = 1
        
        # `catchment` object
        catchment = gpd.GeoDataFrame(pd.concat([cat, cst]))
    else:
        catchment = cat
        
    # assign crs
    catchment.set_crs(epsg=crs, inplace=True)
    
    # sort based on `cat_col_id` and reset_index
    catchment.sort_values(by=cat_col_id, axis='index', inplace=True)
    catchment.reset_index(drop=True, inplace=True)
    
    return catchment


def box_to_geometry(
    lon_min: float = None,
    lat_min: float = None,
    lon_max: float = None,
    lat_max: float = None,
    lims_box: list = [],
    crs: int = 4326,
    **kwargs,
) -> gpd.GeoDataFrame:
    '''Converts a latitude and longitude bounds to a geopandas
    GeoDataFrame object
    
    Parameters
    ----------
    lon_min: float, optional [defaults to None]
        Minimum longitude value
    lat_min: float, optional [defaults to None]
        Minimum latitude value
    lon_max: float, optional [defaults to None]
        Maximum longitude value
    lon_min: float, optional [defaults to None]
        Maximum latitude value
    box: list of floats, optional [defaults to [] ]
        List of spatial bounds in the ``[lon_min, lat_min, lon_max, lat_max]`` 
        form
    crs: int, optional [defaults to 4326]
        Coordinate Reference System EPSG numeber, defaults to
        WGS84 [EPSG:4326]
        
    Returns
    -------
    geom: geopandas.GeoDataFrame
        a GeoDataFrame object containing the geometry of the 
        specified spatial bounds
    '''
    
    if lon_min and lat_min and lon_max and lat_max is not None:
        geometry = box(lon_min, lat_min, lon_max, lat_max, ccw=True)
    elif any(lims_box):
        geometry = box(*lims_box)
    else:
        raise ValueError("Either the spatial limits or a list of them must be provided")
    
    return gpd.GeoDataFrame(index=[0], crs=f'epsg:{crs}', geometry=[geometry])


def prepare_riv(
    riv: gpd.GeoDataFrame,
    riv_cols: Dict[str, str],
    cat: gpd.GeoDataFrame,
    cat_cols: Dict[str, str],
    outlet_val: int = -9999,
    riv_na_val: int = 1,
    crs: int = 4326,
    *args,
    **kwargs,
) -> gpd.GeoDataFrame:
    '''Preparing the `river` object
    
    Parameters
    ----------
    riv: geopandas.GeoDataFrame
        The river network object containing river segments
    riv_cols: dict
        A dictionary containing 'id', 'next_id', keys and values 
        corresponding to the river segement IDs and their downstream
        river segments, respectively. Also, other keys includes 'slope',
        'length', 'length_direct'
    cat: geopandas.GeoDataFrame
        The catchment topology object derived from `prepare_cat`
        function
    cat_cols: dict
        A dictionary containing 'id', 'hillslope' keys and values
        corresponding to the catchment IDs and also the flag indicating
        whether the catchment is a coastal hillslope one (1) or not (0)
    outlet_val: int, optional [defaults to -9999]
        Value being assigned to the 'next_id' cell of the each
        river segments indicating the segment is an outlet for the network;
        before any subsetting, if any coastal hillslope segments exist, they
        will only be set to the outlet value.
    riv_na_val: int, optional [defaults to 1]
        An integer value filling the geopandas.GeoDataFrame `river` object
        to replace `NaN`
    crs: int, optional [defaults to 4326]
        The Coordination Reference System EPSG number
    
    Returns
    -------
    river: geopandas.GeoDataFrame
        The prepared river network topology GeoDataFrame object        
    '''
    # necessary definitions
    hillslope_id = cat_cols.get('hillslope')
    cat_col_id = cat_cols.get('id')
    cat_col_geom = cat_cols.get('geom')
    riv_col_id = riv_cols.get('id')
    riv_col_next_id = riv_cols.get('next_id')
    riv_col_len = riv_cols.get('length')
    riv_col_len_dir = riv_cols.get('length_direct')
    riv_col_slope = riv_cols.get('slope')
    
    
    # necessary assertions
    assert isinstance(riv, gpd.GeoDataFrame), "`riv` object must be of type geopandas.GeoDataFrame"
    assert isinstance(riv_cols, dict), "`riv_cols` object must be of type dict"
    assert isinstance(cat, gpd.GeoDataFrame), "`cat` object must be of type geopandas.GeoDataFrame"
    
    # `id` and `next_id` keys must be included in the `riv_cols` dictionary
    if not 'id' and 'next_id' in riv_cols:
        raise ValueError("`id` and `next_id` must be included in the `riv_cols` dictionary")
    
    # extract the hillslope column from `cat` and assign `id` as index
    hills = cat.set_index(cat_col_id)
    hills = hills[hillslope_id]
    
    # add the hillslope flag from `cat`
    # `cat` has necessary greater than or equal number of items than `riv`
    river = gpd.GeoDataFrame(riv.join(other=hills, on=riv_col_id, how='outer'), 
                             geometry=cat_col_geom)
    
    # resetting index of `river`
    river.reset_index(drop=True, inplace=True)
    
    # necessary columns being chosen for downcasting of dtypes
    # and other necessary assignments
    col_list = list(river.columns)
    for col in [cat_col_id, cat_col_geom]:
        col_list.remove(col)
    
    # fill NAs with `riv_na_val` values in `rivers`
    river.loc[river[hillslope_id] == 1, col_list] = riv_na_val
    # assign `outlet_val` to the `hillslope_id` river segments
    river.loc[river[hillslope_id] == 1, riv_col_next_id] = outlet_val
    # fix cicular segemnts, in other words, those whose 'id' and
    # 'next_id's are the same
    river.loc[river[riv_col_id] == river[riv_col_next_id], riv_col_next_id] = outlet_val
    # fix zero and negative values for fields that are not acceptable
    river_sec = [riv_col_len, riv_col_len_dir, riv_col_slope]

    # remove possible `None` values from `river_sec`
    while None in river_sec:
        river_sec.remove(None)

    # creating `river_sec_df` 
    river_sec_df = river.loc[:, river_sec]
    river_sec_df.where(~(river_sec_df <= 0), riv_na_val, inplace=True)
    river_sec_df.where(~(river_sec_df.isnull()), riv_na_val, inplace=True)
    river.loc[:, river_sec] = river_sec_df
    
    
    # downcast dtype of necessary columns to integer, if possible
    fcols = river[col_list].select_dtypes('float').columns
    river[fcols] = river[fcols].apply(pd.to_numeric, downcast='integer')
    
    # sort based on `cat_col_id` and reset_index
    river.sort_values(by=riv_col_id, axis='index', inplace=True)
    river.reset_index(drop=True, inplace=True)
    
    # assign crs
    river.set_crs(epsg=crs, inplace=True)
    
    return river


def intersect_topologyy(
    cat: gpd.GeoDataFrame,
    cat_cols: Dict[str, str],
    riv: gpd.GeoDataFrame,
    riv_cols: Dict[str, str],
    shapefile: gpd.GeoDataFrame = None,
    outlet_id: Optional[Union[int, Sequence[int]]] = None,
    outlet_val: Optional[int] = -9999,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    '''Intesecting `cat` and `riv` with a given `shapefile`
    or a specified `outlet_id` of interest.

    Parameters
    ----------
    cat: geopandas.GeoDataFrame
        The catchment GeoDataFrame output of `prepare_cat` function
    cat_cols: dict
        A dictionary with 'id' key corresponding to the `cat` ID
        column
    riv: geopandas.GeoDataFrame
        The river GeoDataFrame output of `prepare_riv` function
    riv_cols: int
        A dictionary with 'id' and 'next_id' keys corresponding to
        the `riv` ID and next downstream ID column
    shapefile: geopandas.GeoDataFrame, optional [defaults to None]
        A GeoDataFrame specifying the limits of the river network
        topology of interest
    outlet_id: int, or sequence of ints, optional [defaults to None]
        the ID or list of IDs specifying the most downstream river segment
        to be specified as the outlet of the river network topology
        in addition to any non-contributing subbasins
    outlet_val: int, optional [defailts to -9999]
        An integer being set to the most downstream river segments

    Returns
    -------
    cat_clipped: geopandas.GeoDataFrame
        The clipped catchments object
    riv_clipped: geopandas.GeoDataFrame
        The clipped rivers object
    '''
    # necessary definitions
    cat_col_id = cat_cols.get('id')
    riv_col_id = riv_cols.get('id')
    riv_col_next_id = riv_cols.get('next_id')

    if shapefile is not None:
        # check `shapefile` dtype
        assert isinstance(shapefile, gpd.GeoDataFrame), "`shapefile` must be of type geopandas.GeoDataFrame"
        # intersection with `cat`
        upstream_ids = cat.overlay(shapefile,
                                   how='intersection',
                                   keep_geom_type=True)[cat_col_id]

    elif outlet_id is not None:
        # if it is a single ID, make a set out of it
        outlet_id = set(outlet_id)

        # check if outlet_id is included in the `riv` IDs
        assert riv[riv_col_id].isin(outlet_id).any(), "`outlet_id` must be chosen from segments included in `riv`"

        # find upstream segments
        upstream_ids = set()
        for element in outlet_id:
            upstream_ids.update(find_upstream(gdf=riv,
                                              target_id=element,
                                              main_id=riv_col_id,
                                              ds_main_id=riv_col_next_id))
    else:
        raise NotImplementedError("Either `shapefile` or `outlet_id` must be specified")

    # index `cat` and `riv` and return `cat_clipped` and `riv_clipped`
    cat_clipped = cat.loc[cat[cat_col_id].isin(upstream_ids), :].copy()
    riv_clipped = riv.loc[riv[riv_col_id].isin(upstream_ids), :].copy()

    # assign `outlet_val` to the segments without downstream segments
    riv_clipped.loc[~riv_clipped[riv_col_next_id].isin(riv_clipped[riv_col_id]), riv_col_next_id] = outlet_val

    return cat_clipped, riv_clipped
    

def flag_ncaalg(
    gdf1: gpd.GeoDataFrame,
    gdf2: gpd.GeoDataFrame,
    threshold: float = 0.1,  # Threshold set to 10% by default
    output_path: str = None
) -> gpd.GeoDataFrame:
    """
    Improve spatial analysis with `assign_intersection_flag_gdf_sindex` function.

    Introduce the `assign_intersection_flag_gdf_sindex` function to efficiently identify
    hydrological polygons in a GeoDataFrame that intersect with non-contributing area polygons
    beyond a specified threshold percentage. This enhancement uses spatial indexing for optimized
    performance, significantly reducing computation time for large datasets. The function adds
    an "iwf" column to the primary GeoDataFrame, initially marking all polygons with a value of 1.
    Polygons representing non-contributing areas that meet the intersection threshold are then
    updated to a value of 2. This update facilitates the distinction of critical hydrological features
    based on their spatial relationship with non-contributing zones, enabling more nuanced environmental
    and hydrological analyses.

    Parameters
    gdf1 : gpd.GeoDataFrame
        The first GeoDataFrame.
    gdf2 : gpd.GeoDataFrame
        The second GeoDataFrame.
    threshold : float, optional
        The threshold for considering an intersection significant, as a fraction of
        the first GeoDataFrame's polygon area (default is 0.1 for 10%).
    output_path : str, optional
        Path where the modified first GeoDataFrame should be saved. If None, the file is not saved.

    Returns
    -------
    gpd.GeoDataFrame
        The modified GeoDataFrame of the first GeoDataFrame with the 'iwf' column added.
    """
    # Initialize the 'iwf' column to 1 for all rows in gdf1 as default
    gdf1['iwf'] = 1
    
    # Create spatial index for gdf2
    spatial_index = gdf2.sindex
    
    # Iterate over gdf1 using spatial indexing to find potential intersections
    for index, row in gdf1.iterrows():
        # Use spatial index to find potential intersections
        possible_matches_index = list(spatial_index.query(row['geometry'], predicate='intersects'))
        if not possible_matches_index:
            continue  # No intersections, move to next row
        
        # Filter possible matches for actual intersection
        possible_matches = gdf2.iloc[possible_matches_index]
        actual_intersections = possible_matches[possible_matches.intersects(row['geometry'])]
        
        # Calculate area fractions for actual intersections
        for _, match in actual_intersections.iterrows():
            intersection_area = row['geometry'].intersection(match['geometry']).area
            area_fraction = intersection_area / row['geometry'].area
            if area_fraction > threshold:
                gdf1.at[index, 'iwf'] = 2  # Update 'iwf' to 2 for significant intersections
                break  # Break after finding the first significant intersection
                
    # Save the modified gdf1 to a new shapefile if an output path is provided
    if output_path is not None:
        gdf1.to_file(output_path)
    
    return gdf1