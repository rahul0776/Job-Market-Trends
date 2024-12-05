import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('./app/nerual_net_Atharva.sav', 'rb') as file:
    loaded_model = pickle.load(file)


columns = ['Year', 'Education.Degrees.Bachelors', 'Education.Degrees.Doctorates',
           'Education.Degrees.Masters', 'Education.Degrees.Professionals',
           'Employment.Employer Type.Business/Industry',
           'Employment.Employer Type.Educational Institution',
           'Employment.Employer Type.Government',
           'Employment.Reason Working Outside Field.Career Change',
           'Employment.Reason Working Outside Field.Family-related',
           'Employment.Reason Working Outside Field.Job Location',
           'Employment.Reason Working Outside Field.No Job Available',
           'Employment.Reason Working Outside Field.Other',
           'Employment.Reason Working Outside Field.Pay/Promotion',
           'Employment.Reason Working Outside Field.Working Conditions',
           'Employment.Reason for Not Working.Family',
           'Employment.Reason for Not Working.Layoff',
           'Employment.Reason for Not Working.No Job Available',
           'Employment.Reason for Not Working.No need/want',
           'Employment.Reason for Not Working.Student',
           'Employment.Work Activity.Accounting/Finance/Contracts',
           'Employment.Work Activity.Applied Research',
           'Employment.Work Activity.Basic Research',
           'Employment.Work Activity.Computer Applications',
           'Employment.Work Activity.Design',
           'Employment.Work Activity.Development',
           'Employment.Work Activity.Human Resources',
           'Employment.Work Activity.Managing/Supervising People/Projects',
           'Employment.Work Activity.Other',
           'Employment.Work Activity.Productions/Operations/Maintenance',
           'Employment.Work Activity.Professional Service',
           'Employment.Work Activity.Qualitity/Productivity Management',
           'Employment.Work Activity.Sales, Purchasing, Marketing',
           'Employment.Work Activity.Teaching', 'Education.Major_Animal Sciences',
           'Education.Major_Anthropology and Archeology',
           'Education.Major_Area and Ethnic Studies',
           'Education.Major_Atmospheric Sciences and Meteorology',
           'Education.Major_Biochemistry and Biophysics',
           'Education.Major_Biological Sciences', 'Education.Major_Botany',
           'Education.Major_Chemical Engineering', 'Education.Major_Chemistry',
           'Education.Major_Civil Engineering',
           'Education.Major_Computer Science and Math',
           'Education.Major_Criminology', 'Education.Major_Earth Sciences',
           'Education.Major_Economics', 'Education.Major_Electrical Engineering',
           'Education.Major_Environmental Science Studies',
           'Education.Major_Food Sciences and Technology',
           'Education.Major_Forestry Services',
           'Education.Major_Genetics, Animal and Plant',
           'Education.Major_Geography', 'Education.Major_Geology',
           'Education.Major_History of Science',
           'Education.Major_Information Services and Systems',
           'Education.Major_International Relations',
           'Education.Major_Linguistics',
           'Education.Major_Management & Administration',
           'Education.Major_Mechanical Engineering',
           'Education.Major_Nutritional Science',
           'Education.Major_OTHER Agricultural Sciences',
           'Education.Major_OTHER Geological Sciences',
           'Education.Major_OTHER Physical and Related Sciences',
           'Education.Major_Oceanography', 'Education.Major_Operations Research',
           'Education.Major_Other Engineering',
           'Education.Major_Pharmacology, Human and Animal',
           'Education.Major_Philosophy of Science',
           'Education.Major_Physics and Astronomy',
           'Education.Major_Physiology, Human and Animal',
           'Education.Major_Plant Sciences',
           'Education.Major_Political Science and Government',
           'Education.Major_Political and related sciences',
           'Education.Major_Psychology', 'Education.Major_Public Policy Studies',
           'Education.Major_Sociology', 'Education.Major_Statistics',
           'Education.Major_Zoology, General']

output_classes =['<50,000', '50,000 - 100,000', '100,000<']

def process_and_predict(new_data, model, columns, classes):
    """
    Process a new data point, align it to the training data structure, predict using the model,
    and process the output for interpretation.

    Parameters:
    - new_data (pd.DataFrame): New data point(s) as a DataFrame.
    - model: Trained model with a `predict` method.
    - columns (list): Column structure of the training data (one-hot encoded features).
    - classes (list): List of class labels corresponding to the output (e.g., ['Low', 'Medium', 'High']).

    Returns:
    - dict: A dictionary with predicted class and probabilities.
    - pd.DataFrame: Aligned data ready for prediction.
    """
    # Step 1: Align the new data to the training data structure
    aligned_data = new_data.reindex(columns=columns, fill_value=0)

    # Step 2: Predict using the model
    predictions = model.predict(aligned_data)

    # Step 3: Process the output
    result_list = []
    for idx, prediction in enumerate(predictions):
        # Identify the predicted class
        predicted_index = np.argmax(prediction)
        predicted_class = classes[predicted_index]

        # Extract class probabilities
        class_probabilities = {classes[i]: prediction[i] for i in range(len(classes))}

        # Prepare user-friendly output
        result = {
            "Predicted Class": predicted_class,
            "Probabilities": class_probabilities,
            "Aligned Data": aligned_data.iloc[idx].to_dict()
        }
        result_list.append(result)

    return result_list, aligned_data

def visualize_prediction(probabilities, classes):
    """
    Visualize the class probabilities as a bar chart.

    Parameters:
    - probabilities (dict): Dictionary of class probabilities.
    - classes (list): List of class labels.
    """
    plt.bar(classes, probabilities.values(), alpha=0.7)
    plt.title("Predicted Class Probabilities")
    plt.ylabel("Probability")
    plt.xlabel("Classes")
    plt.ylim(0, 1)
    plt.show()


def app():

    st.title(" Expected Salary range Predictor")

    st.sidebar.header("Input Features")
    year = st.sidebar.number_input("Year", min_value=2000, max_value=2025, step=1, value=2022)
    education_level = st.sidebar.selectbox(
        "Education Level", ["Bachelors", "Masters", "Doctorates", "Professionals"]
    )

    employer_type = st.sidebar.selectbox(
        "Employer Type", ["Business/Industry", "Educational Institution", "Government"]
    )

    reason_outside_field = st.sidebar.selectbox(
        "Reason for Working Outside Field", [
            "", "Career Change", "Family-related", "Job Location", "No Job Available", "Other"
        ]
    )

    input_data = pd.DataFrame([{  # Example one-hot encoding structure
        'Year': year,
        'Education.Degrees.Bachelors': int(education_level == "Bachelors"),
        'Education.Degrees.Masters': int(education_level == "Masters"),
        'Education.Degrees.Doctorates': int(education_level == "Doctorates"),
        'Education.Degrees.Professionals': int(education_level == "Professionals"),
        'Employment.Employer Type.Business/Industry': int(employer_type == "Business/Industry"),
        'Employment.Employer Type.Educational Institution': int(employer_type == "Educational Institution"),
        'Employment.Employer Type.Government': int(employer_type == "Government"),
        'Employment.Reason Working Outside Field.Career Change': int(reason_outside_field == "Career Change"),
        'Employment.Reason Working Outside Field.Family-related': int(reason_outside_field == "Family-related"),
        'Employment.Reason Working Outside Field.Job Location': int(reason_outside_field == "Job Location"),
        'Employment.Reason Working Outside Field.No Job Available': int(reason_outside_field == "No Job Available"),
        'Employment.Reason Working Outside Field.Other': int(reason_outside_field == "Other"),
    }])

    if st.sidebar.button("Predict"):
            results, aligned_data = process_and_predict(input_data, loaded_model, columns, output_classes)
            for result in results:
                st.write(f"Predicted Salary Range: {result['Predicted Class']}")
                st.write("Class Probabilities:")
                st.json(result['Probabilities'])
                visualize_prediction(result['Probabilities'], output_classes)

if __name__ == "__main__":
    app()
