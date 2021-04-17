import  os 
import pickle
import  pandas as pd
from csv import writer

datapath = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))

df = pd.read_csv(datapath + '/IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv')
df.drop('Unnamed: 0', axis=1, inplace=True)

df.rename(columns = { 'Data' : 'Date',
                      'Industry Sector' : 'Industry_Sector', 
                      'Accident Level': 'Accident_Level',
                      'Countries' : 'Country',
                      'Genre' : 'Gender',
                      'Potential Accident Level' : 'Potential_Accident_Level',
                      'Employee or Third Party' : 'Employee_Type', 
                      'Critical Risk' : 'Critical_Risk'}, inplace = True)

def predict_model(description):        
    cv=pickle.load(open(datapath + '/tranform.pkl','rb'))
    clf = pickle.load(open(datapath + '/nlp_chatbot.sav','rb'))

    x = [description]
    vect = cv.transform(x).toarray()
    prediction = clf.predict(vect)
    return prediction

def get_unique_values():
    return {col: df[col].unique().tolist() for col in df.columns if (col !='Description') and (col !='Date') and (col != "Potential_Accident_Level")}

def write_csv(country, local, industry_sector, gender, employee_type, critical_risk, accident_level, description, prediction):

    List=[country, local, industry_sector, gender, employee_type, critical_risk, accident_level, description, prediction]

    try:
        with open(datapath + '/Test.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(List)
            f_object.close()
    except Exception as err:
        print(err)        