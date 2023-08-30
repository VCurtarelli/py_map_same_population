import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.path as mpltPath
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "base",
    type=str,
    default="SC-Florianópolis",
    help="State and city for base population and shape. Ex: SCFlorianópolis, SC-Florianópolis or 'SC-Braço do norte'")
parser.add_argument(
    "centroid",
    type=str,
    default="BA-Correntina",
    help="State and city for centroid. Ex: SCFlorianópolis, SC-Florianópolis or 'SC-Braço do norte'"
)


def find_centroid(geo_data_city):
    """
    Finds the centroid of city, given its border in vertices.

    Args:
        geo_data_city: List
            Coordinates of border of city.

    Returns:
        Coordinates of centroid of city.

    """
    coord = geo_data_city
    A = 0
    N = len(coord)
    coord.append(coord[0])
    for i in range(N):
        xi_0 = coord[i][0]
        yi_0 = coord[i][1]
        xi_1 = coord[i + 1][0]
        yi_1 = coord[i + 1][1]

        A += 1 / 2 * (xi_0 + xi_1) * (yi_1 - yi_0)
    Cx = 0
    Cy = 0
    for i in range(N):
        xi_0 = coord[i][0]
        yi_0 = coord[i][1]
        xi_1 = coord[i + 1][0]
        yi_1 = coord[i + 1][1]

        Cx += 1 / (6 * A) * (xi_0 + xi_1) * (xi_0 * yi_1 - xi_1 * yi_0)
        Cy += 1 / (6 * A) * (yi_0 + yi_1) * (xi_0 * yi_1 - xi_1 * yi_0)

    Cx = Cx
    Cy = Cy
    return [Cx, Cy]


def load_geodata(json_filename, pkl_filename):
    """
    Loads geospatial data from a .json file, converts do a Pandas dataframe, and saves as a .pkl file. If the .pkl file
    already exists, loads it instead.

    Args:
        json_filename: str
            Filename of .json file.
        pkl_filename: str
            Filename of .pkl file.

    Returns: Pandas dataframe with the loaded data.

    """
    if os.path.exists(pkl_filename):
        geo_data = pd.read_pickle(pkl_filename)
    else:
        geo_data = pd.read_json(json_filename,
                                encoding='UTF-8')
        new_geo_data = pd.DataFrame(columns=['id', 'Município', 'Coord', 'Center'])
        for i in range(len(geo_data)):
            new_item = {
                'id': int(geo_data['features'][i]['properties']['id']),
                'Município': geo_data['features'][i]['properties']['name'],
                'Coord': geo_data['features'][i]['geometry']['coordinates'][0],
                'Center': find_centroid(geo_data['features'][i]['geometry']['coordinates'][0])}
            new_geo_data.loc[len(new_geo_data)] = new_item

        geo_data = new_geo_data
        geo_data.to_pickle(pkl_filename)

    return geo_data


def load_popdata(csv_filename, pkl_filename):
    """
    Loads populational data from a .csv file, converts do a Pandas dataframe, and saves as a .pkl file. If the .pkl
    file already exists, loads it instead.

    Args:
        csv_filename: str
            Filename of .csv file.
        pkl_filename: str
            Filename of .pkl file.

    Returns: Pandas dataframe with the loaded data.

    """
    if os.path.exists(pkl_filename):
        pop_data = pd.read_pickle(pkl_filename)
    else:
        pop_data = pd.read_csv(csv_filename,
                               sep=';',
                               decimal='.')
        pop_data.drop(columns=pop_data.columns[-1], axis=1, inplace=True)
        pop_data.drop(columns=['IBGE', 'Porte', 'Região', 'Capital'])
        pop_data.rename(columns={'ConcatUF+Mun': 'UFMun',
                                 'Município': 'Mun',
                                 'População 2010': 'Pop',
                                 'IBGE7': 'id'}, inplace=True)
        pop_data = pop_data[['id', 'UFMun', 'UF', 'Mun', 'Pop']]
        pop_data.to_pickle(pkl_filename)
    return pop_data


def main(BC_state_city: str, CC_state_city: str) -> None:
    """
    Receives the desired base-city, from which to use the population and shape; and center-city, around which to draw
    the region with same population.

    Args:
        BC_state_city: str
            Base state and city to use for population and shape.
        CC_state_city: str
            State and city to use for the centroid.
    """

    # Loads data
    pop_data_filename_org = 'files/fontes/Lista_Municípios_com_IBGE_Brasil_Versao_CSV.csv'
    pop_data_filename_pkl = 'files/popdata.pkl'
    pop_data = load_popdata(pop_data_filename_org, pop_data_filename_pkl)

    geo_data_filename_json = 'files/fontes/geojs-100-mun.json'
    geo_data_filename_pkl = 'files/geodata.pkl'
    geo_data = load_geodata(geo_data_filename_json, geo_data_filename_pkl)

    assert BC_state_city in pop_data["UFMun"].values, f"State-city provided for reference '{BC_state_city}' is not in database"
    assert CC_state_city in pop_data["UFMun"].values, f"State-city provided for centroid '{CC_state_city}' is not in database"

    # Relevant info from base-city
    BC_pop_data = pop_data[pop_data['UFMun'] == BC_state_city]
    BC_pop = BC_pop_data['Pop'].item()
    BC_id = BC_pop_data['id'].item()

    BC_geo_data = geo_data[geo_data['id'] == BC_id]

    BC_pol = BC_geo_data['Coord'].item()
    BC_cnt = BC_geo_data['Center'].item()

    # Relevant info from center-city
    CC_pop_data = pop_data[pop_data['UFMun'] == CC_state_city]
    CC_id = CC_pop_data['id'].item()

    CC_geo_data = geo_data[geo_data['id'] == CC_id]
    CC_cnt = CC_geo_data['Center'].item()

    # Calculates size of region around center-city, with similar shape and population as base-city
    scl_lower = 0
    scl_upper = 4
    pop_flag = False

    while True:
        BC_pol_over_CC = []
        id_contained_cities = []
        scale = (scl_lower + scl_upper) / 2

        # New polygon over center-city
        for idx, vtx in enumerate(BC_pol):
            oX = vtx[0]
            oY = vtx[1]
            coX = BC_cnt[0]
            coY = BC_cnt[1]
            cnX = CC_cnt[0]
            cnY = CC_cnt[1]
            nX = (oX - coX) * scale + cnX
            nY = (oY - coY) * scale + cnY
            BC_pol_over_CC.append([nX, nY])
        BC_pol_over_CC.append(BC_pol_over_CC[-1])

        # Calculate population inside new polygon
        path = mpltPath.Path(BC_pol_over_CC)
        pop_sum = 0
        for i in range(len(pop_data)):
            IC_id = pop_data.loc[i, 'id']
            IC_pop_data = pop_data[pop_data['id'] == IC_id]
            IC_pop = IC_pop_data['Pop'].item()
            IC_geo_data = geo_data[geo_data['id'] == IC_id]
            if IC_geo_data.empty:
                continue
            IC_cnt = IC_geo_data['Center'].item()
            contains = path.contains_points(np.array(IC_cnt).reshape(1, -1))

            if any(contains):
                pop_sum += IC_pop
                id_contained_cities.append(IC_id)

        # Check if population inside new polygon is greater than base-city's population
        if pop_sum < BC_pop and pop_flag is False:
            scl_upper = 2 * scl_upper
        else:
            pop_flag = True
            if pop_sum > BC_pop:
                scl_lower = scl_lower
                scl_upper = scale
            elif pop_sum < BC_pop:
                scl_lower = scale
                scl_upper = scl_upper
        if (1 - 0.02) * BC_pop < pop_sum < (1 + 0.02) * BC_pop:
            break

    # Draw map with base-city and contained-cities highlighted
    plt.figure()
    for i in range(len(pop_data)):
        IC_id = pop_data.loc[i, 'id']
        IC_geo_data = geo_data[geo_data['id'] == IC_id]
        if IC_geo_data.empty:
            continue
        IC_pol = IC_geo_data['Coord'].item()
        xs, ys = zip(*IC_pol)
        if IC_id == BC_id:
            plt.fill(xs, ys, c='b', linewidth=0.1)
        if IC_id not in id_contained_cities:
            plt.plot(xs, ys, c='k', linewidth=0.1)
        else:
            plt.fill(xs, ys, c='r', linewidth=0.1)
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args.base.replace("-", ""), args.centroid.replace("-", ""))