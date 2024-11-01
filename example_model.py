import google.generativeai as genai
import google.oauth2.credentials
import os
import json
import google.ai.generativelanguage as glm
from google.auth import default
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import google.auth
from google.auth.transport.requests import Request

SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/generative-language.retriever',
    'https://www.googleapis.com/auth/youtube.force-ssl'
]

def get_credentials():
    try:
        creds, project = default(scopes=SCOPES)
        return creds
    except Exception as e:
        SERVICE_ACCOUNT_FILE = 'service_account_key.json'
        creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return creds
    
retriever_service_client = glm.RetrieverServiceClient(credentials=get_credentials())
generative_service_client = glm.GenerativeServiceClient(credentials=get_credentials())

def create_youtube_service():
    youtube = build('youtube', 'v3',credentials= get_credentials())
    return youtube

def search_youtube_video(query):
    youtube = create_youtube_service()
    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        maxResults=1
    ).execute()

    if 'items' in search_response and len(search_response['items']) > 0:
        video_id = search_response['items'][0]['id']['videoId']
        return video_id
    else:
        return None

def query_corpus(corpus_resource_name, user_query, results_count=5):
    query_request = glm.QueryCorpusRequest(
        name=corpus_resource_name,
        query=user_query,
        results_count=results_count
    )
    query_response = retriever_service_client.query_corpus(query_request)
    return query_response

def generate_answer(corpus_resource_name, user_query, answer_style="EXTRACTIVE"):
    content = glm.Content(parts=[glm.Part(text=user_query)])
    retriever_config = glm.SemanticRetrieverConfig(source=corpus_resource_name, query=content)
    generate_answer_request = glm.GenerateAnswerRequest(
        model="models/aqa",
        contents=[content],
        semantic_retriever=retriever_config,
        answer_style=answer_style,
        temperature=0.2
    )
    aqa_response = generative_service_client.generate_answer(generate_answer_request)
    return aqa_response

def handle_user_query(corpus_resource_name, user_query, base64_image, results_count=5):
    aqa_response = generate_answer(corpus_resource_name, user_query)
    print("HEre's the full AQA response: \n", aqa_response)
    answerable_probability = aqa_response.answerable_probability
    if answerable_probability <= 0.9:
        print("AQA Probability low")
        model = genai.GenerativeModel(model_name='gemini-1.5-flash',
                                       generation_config={
                                            "temperature": 0.3,
                                            "top_p": 1,
                                            "top_k": 40,
                                            "max_output_tokens": 2048,
                                            "response_mime_type": "application/json"
                                        },
                                        safety_settings=[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}])
        response = model.generate_content([user_query, base64_image])
        return response.text

    try:
        answer_content = aqa_response.answer.content.parts[0].text
    except AttributeError:
        print("Error: The expected field was not found in the AQA response.")
        return None

    return answer_content
    
# MAKE SURE YOU ARE SENDING IN THE RIGHT STUFF IF YOU ARE USING MY CODE!!!!!
def generate_greenwashing_response(spoken_text, base64_image):
    text_prompt = f"""
    Answer based on the data based on user prompt: {spoken_text}


    Give it in this JSON format:
    {{
    "Response": "[Your response here]",
    "Video_Suggestion": "<add a search query to search for a cool youtube video based on this>"
    "Keyword": "[Give a google search keyword based on what the user is asking]"
    }}
    """

    corpus_resource_name = "corpora/my-corpus-94qlvnd3wanj"

    try:
        # Handle the user query with RAG
        rag_response = handle_user_query(corpus_resource_name, text_prompt, base64_image)

        if rag_response is None:
            return {'error': "Query response structure is unexpected."}

        try:
            # we try to parse the response
            text_analysis_result = json.loads(rag_response)
            response = text_analysis_result.get("Response", "No response generated.")
            keyword = text_analysis_result.get("Keyword", "Unknown")
            video_suggestion = text_analysis_result.get("Video_Suggestion")
        except json.JSONDecodeError:
            # if JSON parsing fails, use 1.5 Flash model as a backup so the user still sees something
            print("Houston we have a problem.")
            model = genai.GenerativeModel(model_name='gemini-1.5-flash',
                                           generation_config={
                                                "temperature": 0.3,
                                                "top_p": 1,
                                                "top_k": 40,
                                                "max_output_tokens": 2048,
                                                "response_mime_type": "application/json"
                                            },
                                            safety_settings=[{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}])
            alt_response = model.generate_content([text_prompt, base64_image])
            # we parse the fallback model's JSON response
            print(alt_response)

            fallback_result = alt_response.text

            text_analysis_result = json.loads(fallback_result)
            response = text_analysis_result.get("Response", "No response generated.")
            keyword = text_analysis_result.get("Keyword", "Unknown")
            video_suggestion = text_analysis_result.get("Video_Suggestion")

        result = {'result': response, 'keyword': keyword}

        # Use YouTube API to get a video ID, regardless of the response source
        if video_suggestion:
            video_id = search_youtube_video(video_suggestion)
            if video_id:
                result['video_suggestion'] = video_id

        return result

    except Exception as e:
        return {'error': str(e)}
