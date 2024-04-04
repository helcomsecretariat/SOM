from som.som_app import process_input_data, build_core_object_model, build_second_object_layer, postprocess_object_layers

def run():

    # Process survey data and read general input
    survey_df, object_data = process_input_data()

    # Build core object model and initialize core object instances
    measure_df = build_core_object_model(survey_df=survey_df, object_data=object_data)

    # Build second object layer model and initialize second object layer instances
    countrybasin_df = build_second_object_layer(measure_df=measure_df, object_data=object_data)

    # Post-process object layers by setting values based on general input
    countrybasin_df = postprocess_object_layers(countrybasin_df=countrybasin_df, object_data=object_data)

if __name__ == "__main__":
    run()