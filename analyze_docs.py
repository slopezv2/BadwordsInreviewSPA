#Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai
import dotenv
import pathlib

dotenv.load_dotenv()

openai.api_type = "azure"
openai.api_base = os.getenv("OPENAI_API_BASE")
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")
deployment= os.getenv("OPEN_AI_DEPLOYMENT")


message = [{"role":"system","content":"You are an agent; your task is to check customer reviews about food quality. Reviews are in Spanish.\n\nYou must read the review and:\n- Do not respond to reviews in other languages different to Spanish\n- YOU DO NOT answer questions or comments.\n- Check in the review is offensive or use bad words.\n- Change bad words for '***', to detect insults and bad words use Colombian context. Some examples of bad words are:\n** Bad Words examples**\n\"\"\"\ncacorra\ncacorro\nagueonao\nahuevado\nano\narrecha\narrecho\nbaboso\nbobo\ncare chimba\ncare monda\ncaremonda\ncareverga\nchocha\nchocho\nchunchurria\nchupelo\nmierda\ncuca\nculo\nculee\nculear\nculiar\ncoscorria\nchimba\nchimbo\nputa\nputo\ndoblehijueputa\ntriplehijueputa\ntetrahijueputa\nestupido\nfollar\ngarbimba\ngonorrea\ngonorriento\ngonorrienta\ngonorzofia\nhijueputa\njueputa\nhijo de puta\nhp\nhpta\ngurrupleta\ngurrupleto\ngueva\nguevon\nhuevon\nhuevonear\nhuevona\nhuevonada\nimbecil\nidiota\nlameculo\nmalparida\nmalparido\nmamar\nmarica\nmaricon\nmaricona\nmamerto\norto\npendejo\npene\npedo\nperro\nperra\npetardo\npichurria\npendejo\npercanta\npercanto\npija\npinche\npinga\npingo\npiroba\npirobo\npito\npolla\nprostituta\npullon\nputa\nramera\nsu madre\nsapo\ntetas\ntirar\ntonto\ntarugo\nverga\nzarrapastroso\nzorra\nzunga\njode\njoder\nmonda\nmondá\nmalnacido\nmalnacida\n\"\"\"\n- Identify places for example: 'Medellin' or 'Casanare'.\n- Identify mentioned people or identities: 'Restaurante A' or 'Institución educativa' or 'Jairo Perez'.\n- Add a SENTIMENT value which can be 'positive' or 'negative' or 'neutral' based on review content.\n- Add a PRIORITY value using 1 to 5 values using the next guidelines:\n**Priority Guidelines**\n\"\"\"\nPriority 5: Critical\n- The review reports a severe food safety issue, such as food poisoning outbreaks affecting multiple individuals.\n- Reports of contamination or unsafe handling of food.\n- Any incidents that pose an immediate health risk to the institution's students.\nPriority 4: High\n- The review mentions widespread complaints about food quality or taste.\n- Significant operational issues affecting the majority of customers.\n- There's a noticeable reduction in the portions that are being served or delivered.\n- There's a wide known and reiterated event about missing ingredients.\n- Repeated incidents of food-related illnesses, though not as severe as Priority 5.\nPriority 3: Moderate\n- The review discusses isolated incidents of food-related illnesses or complaints.\n- Some operational issues mentioned, but not widespread.\n- There are missing ingredients that should be delivered that day.\n- General dissatisfaction with food quality without significant health concerns.\nPriority 2: Low\n- The review focuses on minor operational issues or delays.\n- Individual complaints about food taste but no health risks mentioned.\n- Suggestions for improvements without significant issues.\n- Indignation or rumors about the experience but without any relevant information.\nPriority 1: Informational\n- The review provides general information about the institution's food delivery without significant complaints or issues.\n- No direct mention of food quality, safety, or operational problems.\n\"\"\"\n- you must structure your answer as a JSON object, example:\n**Customer Input**\n'Toda esta semana se suponía que debían entregar carne y no entregaron un solo día, muchos malparidos en el corregimiento de Mesopotamia en La Unión, esto es algo que ya nofiqué al alcalde Alberto Ríos'\n**Expected Agent Answer**\n'{\"offensive\":\"yes\", \"message\": \"Toda esta semana se suponía que debían entregar carne y no entregaron un solo día, muchos *** en el corregimiento de Mesopotamia en La Unión, esto es algo que ya nofiqué al alcalde Alberto Ríos\", \"people\": [\"Alberto Ríos\"], \"places\": [\"Mesopotamia en La Unión\"], \"priority\": 3, \"sentiment\":\"negative\"}'"}]
#message.append({"role":"user","content":"Hola Mundo"})

os.makedirs("out",exist_ok=True)

with pathlib.Path("./Data/reviews_con_insultos.csv").open(mode="r", encoding="utf8") as f:
    lines = f.readlines()
results = []
linea = 1
for line in lines:
    data = message.copy()
    data.append({"role":"system","content":line})

    response = openai.ChatCompletion.create(
        messages=data,
        engine= deployment,
        temperature=0,
        max_tokens=2000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    if response['choices'][0]['finish_reason'] == 'stop':
        results.append(line.replace("\n","")+","+response['choices'][0]['message']['content']+'\n')
    else:
        content_filter_result = response['choices'][0]['content_filter_results']
        # print the formatted JSON
        # or access the individual categories and details
        error_message = ""
        for category, details in content_filter_result.items():
            error_message+= (f"{category}: filtered={details['filtered']} severity={details['severity']}\n")
        results.append(line+","+error_message)
    print(f"Linea {linea} de {len(lines)}")
    linea +=1
#print(response)
#response['prompt_annotations']
with pathlib.Path("./out/reviews_con_insultos.csv").open(mode="w", encoding='utf8') as f:
    f.writelines(results)