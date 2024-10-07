"""
Copyright (c) 2024 Baltic Marine Environment Protection Commission
Copyright (c) 2022 Antti-Jussi Kieloaho (Natural Resources Institute Finland)

LICENSE available under 
local: 'SOM/protect_baltic/LICENSE'
url: 'https://github.com/helcomsecretariat/SOM/blob/main/protect_baltic/LICENCE'
"""

# main package script

import som.som_app as som_app

def run():

    # Process survey data and read general input
    measure_survey_df, pressure_survey_df, object_data = som_app.process_input_data()

    # Build core object model and initialize core object instances
    measure_df = som_app.build_core_object_model(measure_survey_df, pressure_survey_df, object_data)

    # Build second object layer model and initialize second object layer instances
    countrybasin_df = som_app.build_second_object_layer(measure_df=measure_df, object_data=object_data)

    # Post-process object layers by setting values based on general input
    countrybasin_df = som_app.postprocess_object_layers(countrybasin_df=countrybasin_df, object_data=object_data)

if __name__ == "__main__":
    run()

