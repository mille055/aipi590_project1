from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments,pipeline, logging, LlamaTokenizer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,re
import torch
from datasets import load_dataset, Dataset
from trl import SFTTrainer
import pyarrow as pa
import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from google.colab import userdata
import json
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('HUGGINGFACE_TOKEN')

def build_dataset_from_file(file_path, num_test = 200, output_file = '/content/CT_Protocol/data/datacsv031524.xlsx'):
  """
  This function reads an excel file, splits into train/test datasets in a dataframe and saves as a csv file.

  Args:
    filename: The path to the excel file.

  Returns:
    A dataframe from which the train and test datasets can be derived.
  """
  # Read the excel file into a pandas dataframe
  df = pd.read_excel(file_path)

  # Rename columns and identify x and y data
  new_column_names = {'Procedure': 'order', 'Reason for Exam Full': 'indication', 'Previous Procedure Name':'prior_order', 'Contrast Allergy': 'contrast_allergy', 'Allergy Severity': 'allergy_severity', 'Creatinine (mg/dL)':'creatinine', 'Dialysis': 'on_dialysis', 'clinical summary':'clinical_summary', 'Predicted Procedure': 'predicted_order', 'Protocol': 'predicted_protocol', 'Protocol comments':'predicted_comments', 'Accession':'accession'}
  df.rename(columns=new_column_names, inplace=True)
  df.fillna("", inplace=True)
  df['accession'] = df['accession'].astype(str)


  with open(output_file, 'w') as f:
    f.write(df.to_csv(index=False))

  return df

def row_to_json(row, columns):
  """
  This function takes a row of a dataframe and returns a json object with the specified columns.

  Args:
    row: The row of the dataframe.
    columns: A list of columns to include in the json object.

  Returns:
    A json object with the specified columns.
  """
  json_obj = {}
  for column in columns:
    value = row[column]
    if column == "predicted_comments" and not isinstance(value, list):
            # Attempt to convert a string representation of a list into an actual list
            # Only do this if the value is not already a list
            try:
                # This handles the case where the value is a string representation of a list
                value = json.loads(value.replace("'", '"'))
            except:
                # If there's an error (e.g., value is not a valid list string), set to an empty list
                value = []
    json_obj[column] = value
  #print(type(json_obj))
  return json_obj



def build_prompt_question(row, prompt_instruction=prompt_instruction2):

  prompt_question = 'Order: ' + row['order'] + '\n' + \
    'Prior Order: ' + row['prior_order'] + '\n' + \
    'Reason for Exam: ' + row['indication'] + '\n' + \
    'Contrast Allergy: ' + str(bool(row['contrast_allergy'])) + '\n' + \
    'Allergy severity: ' + row['allergy_severity'] + '\n' + \
    'Creatinine: ' + str(row['creatinine']) + '\n' + \
    'On Dialysis: ' + str(bool(row['on_dialysis'])) + '\n' + \
    'Clinical Summary: ' + row['clinical_summary'] + '\n'

  #print('build_prompt_question sending', prompt_question, type(prompt_question))
  return prompt_question


def build_prompt_question_json(row, columns = ['accession', 'order', 'Reason for Exam', 'prior_order', 'indication',
       'creatinine', 'on_dialysis', 'contrast_allergy', 'allergy_severity', 'clinical_summary']):
  """
  This function takes a row of a dataframe and returns a prompt question.

  Args:
    row: The row of the dataframe.

  Returns:
    A prompt question in json format.
  """

  prompt_question = row_to_json(row, columns)
  #print('build_prompt_question_json sending', prompt_question, type(prompt_question))
  return prompt_question


def build_prompt_answer(row, columns = ["accession", "predicted_order", "predicted_protocol", "predicted_comments"]):
  """
  This function takes a row of a dataframe and returns a prompt answer and prompt answer2.

  Args:
    row: The row of the dataframe.

  Returns:
    A prompt answer and prompt answer2.
  """
  prompt_answer = row_to_json(row, columns)
  if not isinstance(prompt_answer['predicted_comments'], list):
    prompt_answer['predicted_comments'] = [prompt_answer['predicted_comments']]

  return prompt_answer


def create_prompt_dataframe(df):
  """
  This function takes a dataframe and returns a dataframe with the prompt questions and answers.

  Args:
    df: The dataframe to be converted.

  Returns:
    A dataframe with the prompt questions and answers.
  """
  df1 = pd.DataFrame()
  for index, row in df.iterrows():
    prompt_question_text = build_prompt_question(row)
    #print('prompt_question_text', prompt_question_text, type(prompt_question_text))

    prompt_question_json = build_prompt_question_json(row)
    #print('prompt_question_json', prompt_question_json, type(prompt_question_json))


    prompt_answer = build_prompt_answer(row)
    #print('prompt_answer', prompt_answer, type(prompt_answer))

    df1.at[index, 'text'] = prompt_question_text
    df1.at[index, 'prompt_question_json'] = str(prompt_question_json).replace("'", '"')
    df1.at[index, 'labels'] = str(prompt_answer).replace("'", '"')
    #print(df1.head())
  return df1

def extract_and_parse_json(response):
    # Assuming the JSON-like response is always formatted with single quotes,
    # which is invalid JSON format and needs to be replaced with double quotes.
    # Also assuming the JSON-like object is always enclosed in curly braces.
    response = str(response)
   
    try:
        # Correctly handle both empty strings and string-represented empty lists for predicted_comments.
        corrected_response = response.replace("'", '"')

        
        print('Corrected response:', corrected_response)
        
        # Parse the corrected response into a Python dictionary.
        json_data = json.loads(corrected_response)
        print('type of json data', type(json_data))
        return json_data

    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None


def extract_and_parse_json2(response):
    #print('extracting from', response, type(response))
    # Assuming the JSON-like response is always formatted with single quotes,
    # which is invalid JSON format and needs to be replaced with double quotes.
    # Also assuming the JSON-like object is always enclosed in curly braces.
    #response = str(response)

    json_str_match = re.search(r'\{.*\}', response)
    if json_str_match:
      json_str = json_str_match.group(0)

    try:
        # Normalize response by ensuring it uses double quotes.
        normalized_response = json_str.replace("'", '"')
        
        # Correctly handle empty strings for predicted_comments.
        corrected_response = re.sub(r'"predicted_comments":\s*""', '"predicted_comments": []', normalized_response)
        
        # Handle correctly formatted lists and empty lists.
        corrected_response = re.sub(
            r'"predicted_comments":\s*"(\[.*?\])"',
            lambda match: f'"predicted_comments": {match.group(1)}',
            corrected_response)
        # Handle empty strings for predicted_comments by converting them to empty lists.
        corrected_response = re.sub(r'"predicted_comments":\s*""', '"predicted_comments": []', corrected_response)
        
        json_data = json.loads(corrected_response)
    
        # If predicted_comments is a string (due to escaping), parse it separately.
        if isinstance(json_data.get('predicted_comments', ''), str):
            json_data['predicted_comments'] = json.loads(json_data['predicted_comments'])

        return json_data

    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None


def get_response(prompt, pipe):
  sequences = pipe(
    prompt,
    do_sample=True,
    max_new_tokens=100,
    temperature=0.2,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1,
  )
  answer = sequences[0]['generated_text']
  cleaned_answer = answer.replace(prompt, '', 1)

  #print('cleaned_answer is ', cleaned_answer)
  return cleaned_answer



def response_score(json_data, true_data):
    score = 0
    max_score = 10
    # print('true_data type is', type(true_data))
    # print(true_data)
    # print('json_data type is ', type(json_data))


    accession, predicted_order, predicted_protocol, predicted_comments = None, None, None, None


    if json_data and isinstance(json_data, dict):

        accession = true_data.get("accession", "Unknown")
        predicted_order = json_data.get("predicted_order", "")
        predicted_protocol = json_data.get("predicted_protocol", "")
        predicted_comments = json_data.get("predicted_comments", "")

        # 3 points if JSON and has the right keys
        required_keys = ["predicted_order", "predicted_protocol", "predicted_comments"]
        if all(key in json_data for key in required_keys):
            score += 3
            # 5 points if 'predicted_protocol' matches the answer
            if json_data["predicted_protocol"] == true_data["predicted_protocol"]:
                score += 5
            # 1 point each if the 'predicted_order' matches
            if json_data["predicted_order"] == true_data["predicted_order"]:
                score += 1
            # 1 point if 'predicted_comments' match (assuming list comparison)
            if "predicted_comments" in true_data and json_data["predicted_comments"] == true_data["predicted_comments"]:
                score += 1

    else:
      score += 0

    score = (score)/max_score
    # print(score, accession, predicted_order, predicted_protocol, predicted_comments)
    return score, accession, predicted_order, predicted_protocol, predicted_comments


def test_model(df, pipe, prompt_instruction=prompt_instruction):
  overall_score = 0
  results_list = []
  for index, row in df.iterrows():
    # get a response and extract json portion from it
    prompt = prompt_instruction + row['text']
    predicted_answer = get_response(prompt, pipe)
    print('********\n')
    #print('predicted_answer is ', predicted_answer)
    extracted_answer = extract_and_parse_json2(predicted_answer)
    print('********\n')
    print('extracted_answer is ', extracted_answer, type(extracted_answer))

    # get the ground truth answer
    true_answer = row['labels']
    #print('true_answer', true_answer, type(true_answer))
    true_answer_json = json.loads(true_answer.replace("'", '"'))

    print('true answer json:', true_answer_json, type(true_answer_json))

    score, accession, predicted_order, predicted_protocol, predicted_comments = response_score(extracted_answer, true_answer_json)
    overall_score += score
    print(f"Progress: case {index+1} of {len(df)}")
    print(f"score this case: {score}")

    # Accumulate the case results
    results_list.append({
            "index": index,


            "protocol": true_answer_json['predicted_protocol'],
            "predicted_protocol": predicted_protocol,
            "order": true_answer_json['predicted_order'],
            "predicted_order": predicted_order,
            "comments": true_answer_json['predicted_comments'],
            "predicted_comments": predicted_comments,
            "score": score
        })

  results = pd.DataFrame(results_list)
  print(results)
  print(f"Average score: {overall_score/len(df)}")
  results.to_csv('/content/CT_Protocol/data/results.csv', index=False)

  return overall_score/len(df)

def get_dataframes(filename):
    '''
    Returns the full, train, and test dataframes
    '''
    full_df = build_dataset_from_file(filename)
    prompt_df = create_prompt_dataframe(full_df)
    dataset = Dataset(pa.Table.from_pandas(prompt_df))
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=12)
    test_data_df = pd.DataFrame(test_data)
    train_data_df = pd.DataFrame(train_data)

    return prompt_df, train_data_df, test_data_df


def test_model2(df, pipe, prompt_instruction=prompt_instruction2):
  overall_score = 0
  results_list = []
  for index, row in df.iterrows():
    # get a response and extract json portion from it
    prompt = f"""[INST] {prompt_instruction}{row['text']} [/INST]"""
    predicted_answer = get_response(prompt, pipe)
    print('********\n')
    #print('predicted_answer is ', predicted_answer)
    extracted_answer = extract_and_parse_json2(predicted_answer)
    print('********\n')
    print('extracted_answer is ', extracted_answer, type(extracted_answer))

    # get the ground truth answer
    true_answer = row['labels']
    #print('true_answer', true_answer, type(true_answer))
    true_answer_json = json.loads(true_answer.replace("'", '"'))

    print('true answer json:', true_answer_json, type(true_answer_json))

    # #predicted_answer = json.loads(predicted_answer)
    # print('predicted_answer:', predicted_answer, type(predicted_answer))

    score, accession, predicted_order, predicted_protocol, predicted_comments = response_score(extracted_answer, true_answer_json)
    overall_score += score
    print(f"Progress: case {index+1} of {len(df)}")
    print(f"score this case: {score}")

    # Accumulate the case results
    results_list.append({
            "index": index,


            "protocol": true_answer_json['predicted_protocol'],
            "predicted_protocol": predicted_protocol,
            "order": true_answer_json['predicted_order'],
            "predicted_order": predicted_order,
            "comments": true_answer_json['predicted_comments'],
            "predicted_comments": predicted_comments,
            "score": score
        })

  results = pd.DataFrame(results_list)
  print(results)
  print(f"Average score: {overall_score/len(df)}")
  results.to_csv('/content/CT_Protocol/data/results.csv', index=False)

  return overall_score/len(df)




prompt_instruction = '''
Use the inputs to expertly decide on the appropriate protocol for the CT study.  The description of each CT protocol is given in the text below (the name to return for each protocol is in parentheses). For scans in which the order does not match the desired protocol, or if there are other outstanding questions the radiologist needs to resolve (e.g., elevated creatinine above 2.0 mg/dL or history of severe allergic reaction to IV contrast such as trouble breathing, throat swelling, or anaphylaxis, and contrast enhanced scan ordered), then add a comment that will route the case back to the radiologist (comments should be from the list given below.

Also, note it is okay to over-image, but ideally no simpler protocols for studies that require multiphase imaging.
Any acute hemorrhage is ideally scanned by the GI bleed protocol (CT with and without contrast, with arterial and venous phases). 6. Any study specifically ordered to evaluate the chest only (such as a PE study), or the thoracic-abdominal aorta or leg arteries should be routed to the chest/cardiovascular imaging division.
If there is an indeterminate renal or adrenal mass on the prior study, it can be evaluated by adding pre contrast images through the abdomen in addition to the regular protocol (comment: 'precontrast through kidneys or adrenals'). That can be added on without communicating with the ordering provider.

Here is a description of the protocols:

Routine (routine):  Protocol for most patients includes portal venous phase imaging.  No oral contrast is administered by default, but can be added through comments.

Noncontrast (noncon): Protocol when no contrast is indicated or there are contraindications such as severe allergy or elevated creatinine. There are reasons why contrast may not be wanted by the ordering provider even if no contraindications, such as a solitary kidney and mild renal failure.

Dual Liver Protocol (dual liver):  Known or suspected hypervascular liver tumor or suspected metastases from a primary tumor outside the liver for which there are suspected hypervascular liver metastases.  It includes both the hepatic arterial and portal venous phases.  Currently the list of malignancies for this protocol includes neuroendocrine, carcinoid, and thyroid carcinoma.

Cirrhosis Protocol (cirrhosis):  Known or suspected cirrhosis and/or have a known or suspected hepatocellular carcinoma.  It also should be performed in all patients with suspected benign primary liver tumors, such as focal nodular hyperplasia or hepatic adenoma.  This protocol includes acquisitions in the hepatic arterial and the portal venous phases, as well as a delayed phase.

Hepatic Resection Protocol (hepatic resection):  Indicated in all patients anticipating hepatic resection.  It includes thin section images of the liver to include celiac axis and proximal SMA during the hepatic arterial phase and thicker sections through the liver during the portal venous phase.  The images obtained during the hepatic arterial phase undergo volume rendering in 3D.

Radioembolization Protocol (radioembo):  Typically ordered by the Interventional Radiologists for evaluation of a patient following (and possibly before) embolization therapy. This includes arterial and venous phases through the abdomen. The post processing is slightly different than the cirrhosis protocol in that thin images are sent in both arterial and portal venous phases for the 3D lab to assess the vasculature and liver volumes. It should be specifically mentioned in the order, otherwise do not use the protocol.

Pancreas Protocol (pancreas):  Known or suspected pancreatic tumor.  It is occasionally requested in patients with either acute or chronic pancreatitis.  It includes thin section images of the pancreas to include the celiac axis and SMA during the pancreatic phase and images of the liver and pancreas during the venous phase.  Arterial phase images are reconstructed in 3D.

Cholangiocarcinoma Protocol (cholangiocarcinoma):  Known or suspected cholangiocarcinoma.  It includes images of the liver in the portal venous phase as well as through the hilum following a 10 minute delay.  Coronal reformats of the venous phase are included.

Trauma Chest/Abdomen/Pelvis (trauma):  Suspected trauma. Arterial phase imaging through the upper and mid chest followed by portal venous phase imaging of the abdomen and pelvis.  No oral contrast.

Crohns Protocol (crohns):  Evaluation to look for suspected Crohns involvement, but not necessarily for complications of Crohn’s.  If a relatively asymptomatic patient, the patient receives VolumenTM (a negative contrast agent). Enteric arterial phase images of the abdomen and pelvis are acquired, and sagittal and coronal reformats are also included. Similar to other Bowel Protocol except that only a single phase is acquired to minimize radiation dose.

CT Colonography (colon):  For colon cancer and polyp screening.  The patient undergoes bowel prep the night before the scan as well as barium tagging.  Insufflation of CO2 via device after placement of tube into rectum.  Supine and prone imaging, as well as decubitus position if nondistended segments on the two standard positions.

Renal Stone Protocol (renal stone):  Acute flank pain and/or a known or suspected renal calculus.  It includes a low dose noncontrast CT of the kidneys, ureters and bladders with the patient in prone position (unless unable).  Coronal reformats are provided.

Genitourinary Protocol (gu):  Hematuria, known or suspected renal mass, or other indications where evaluation of the ureters is necessary.  It includes low dose non-contrast images of the kidneys only followed by nephrographic phase images of the kidneys and 7 min delayed excretory phase images of the kidneys, ureters, and bladder.  Coronal reformats of the excretory phase are included.

Focused Renal Cyst Protocol (focused renal cyst): For followup of a complicated renal cyst. Pre- and Post-contrast (Nephrographic) imaging through the kidneys to assess for enhancement. There is no imaging of the pelvis and no CT urogram. If in doubt, use the more complete GU Protocol.

RCC Protocol (rcc):  Known renal cell carcinoma, typically in patients who have undergone a nephrectomy or nephron sparing treatment, or possibly for preoperative planning.  It includes noncontrast images of the kidneys followed by a dual liver as described above to assess for metastases.  Coronal reformats in the venous phase are included.

TCC Protocol (tcc):  Intended for patients with known transitional cell carcinoma or bladder cancer, typically who have undergone a cystectomy, focal bladder surgery, or nephroureterectomy.  It is a split bolus technique, with the goal of imaging the patient in both the excretory phase and the portal venous phase in a single acquisition following two boluses of IV contrast prior to scanning.

Renal Donor Protocol (renal donor):  To evaluate the renal anatomy of potential renal donors.  This includes thin section images of the kidneys and renal arteries during the arterial phase and venous phase.  A delayed scout is obtained to document the number of ureters.  A separate 3-D interpretation is performed.

Adrenal Protocol (adrenal): For the evaluation of an indeterminate adrenal mass.  Noncontrast images through the adrenal gland.  The scan is then checked by a physician for further imaging.  If the physician deems necessary, portal venous and 15 minute delayed images through the adrenals follow.

CT Cystogram (cystogram):  To evaluate for bladder injury (typically after pelvic trauma) or a fistula.  Contrast (Renografin 60 diluted in saline) is instilled by gravity through an indwelling Foley catheter.



And for predicted_comments, add comments from the following list if appropriate:
'oral contrast' if oral (otherwise known as PO) contrast has been requested in the indication.
'steroid prep' if has a mild allergy to contrast and a contrasted scan has been requested
'reroute contrast' if the patient has a contraindication to contrast such as elevated creatinine above 2.0 or severe/anaphylaxis contrast allergy
'reroute coverage' if addition body parts may need to be added to the planned procedure
'low pelvis' which extends the caudal range of a CT, particularly for malignancies that may not be fully imaged on our routine protocols which includes vulvar cancer, anal cancer, and perhaps rectal cancer if this is the first time evaluation or there is known recurrent disease low in the pelvis.  Things in the inguinal region or upper thigh or perirectal abscess or perianal fistulous disease may be other possible indications. Perineal infection such as Fournier's gangrene would aslo require this.
'reroute protocol' if there is a complex process such as a fistula that might not be evaluated well on the routine protocols.
'split' indicates the chest order will be read separately from the abdomen and pelvis, which occurs for lung and esophgeal cancer and for patients with lung transplants
'valsalva' indicates the imaging is performed while patient does a valsalva maneuver for evaluation of hernias.

The task is to use the provided information for each patient to return the predicted protocol for the CT study in the form of a json object like this:
{"predicted_order": "CT abdomen pelvis with contrast", "predicted_protocol": "routine", "predicted_comments": ["oral contrast"]}
The response should be the json object and nothing else.

'''


prompt_instruction2 = '''
The task is to use the provided information for each patient to return the predicted protocol for the CT study in the form of a json object like this:
{"predicted_order": "CT abdomen pelvis with contrast", "predicted_protocol": "routine", "predicted_comments": ["oral contrast"]}
The response should be the json object and nothing else.
'''

prompt_instruction3 = '''
The task is to use the provided information for each patient to return the predicted protocol for the CT study in the form of a json object like this:
{"predicted_order": "CT abdomen pelvis with contrast", "predicted_protocol": "routine", "predicted_comments": ["oral contrast"]}
The response should be the json object and nothing else.

And for predicted_comments, add comments from the following list if appropriate:
'oral contrast' if oral (otherwise known as PO) contrast has been requested in the indication.
'steroid prep' if has a mild allergy to contrast and a contrasted scan has been requested
'reroute contrast' if the patient has a contraindication to contrast such as elevated creatinine above 2.0 or severe/anaphylaxis contrast allergy
'reroute coverage' if addition body parts may need to be added to the planned procedure
'low pelvis' which extends the caudal range of a CT, particularly for malignancies that may not be fully imaged on our routine protocols which includes vulvar cancer, anal cancer, and perhaps rectal cancer if this is the first time evaluation or there is known recurrent disease low in the pelvis.  Things in the inguinal region or upper thigh or perirectal abscess or perianal fistulous disease may be other possible indications. Perineal infection such as Fournier's gangrene would aslo require this.
'reroute protocol' if there is a complex process such as a fistula that might not be evaluated well on the routine protocols.
'split' indicates the chest order will be read separately from the abdomen and pelvis, which occurs for lung and esophgeal cancer and for patients with lung transplants
'valsalva' indicates the imaging is performed while patient does a valsalva maneuver for evaluation of hernias.
'''
