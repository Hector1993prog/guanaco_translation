import openai
from datasets import Dataset, DatasetDict
import pickle
import time
from typing import List, Optional
import regex as re


def save_lists_to_files(lists: List, prefix: str, start_index: int = 0) -> None:
    """Save a list of lists to separate files.

    Args:
        lists (List): The list of lists to save.
        prefix (str): The prefix for the file names.
        start_index (int, optional): The starting index for file names. Defaults to 0.

    Returns:
        None
    """

    for i, sublist in enumerate(lists, start=start_index):
        filename = f"{prefix}_{i}.pkl"
        with open(filename, 'wb') as file:
            pickle.dump(sublist, file)

def load_lists_from_files(prefix: str) -> List:
    """Load a list of lists from files with a given prefix.

    Args:
        prefix (str): The prefix for the file names.

    Returns:
        List: A list of loaded lists.
    """

    loaded_lists = []
    i = 0
    while True:
        filename = f"{prefix}_{i}.pkl"
        try:
            with open(filename, 'rb') as file:
                sublist = pickle.load(file)
                loaded_lists.append(sublist)
            i += 1
        except FileNotFoundError:
            break
    return loaded_lists




def get_completion(api_key,prompt, model="gpt-3.5-turbo"):

    '''
    This functions calls the openai.ChatCompletion class and 
    uses to complete the response of the model with your prompts.
    '''
    messages = [{"role": "user", "content": prompt}]
    try:
  #Make your OpenAI API request here
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0, 
            api_key = api_key
        )
        return response.choices[0].message["content"]

    except openai.error.APIError as e:
    #Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an API Error: {e}")
        x= print(f"OpenAI API returned an API Error: {e}")
        with open('openai.error.APIError.txt', 'w', encoding='utf-8') as f:
            f.write(x)
            f.close
        pass
    except openai.error.APIConnectionError as e:
    #Handle connection error here
        print(f"Failed to connect to OpenAI API: {e}")
        x = print(f"Failed to connect to OpenAI API: {e}")
        with open('APIConnectionError.txt', 'w', encoding='utf-8') as f:
            f.write(x)
            f.close
        pass
    except openai.error.RateLimitError as e:
    #Handle rate limit error (we recommend using exponential backoff)
        print(f"OpenAI API request exceeded rate limit: {e}")
        x = print(f"OpenAI API request exceeded rate limit: {e}")
        with open('error.RateLimitError.txt', 'w', encoding='utf-8') as f:
            f.write(x)
            f.close        
        pass


def divide_list_into_samples(input_list: List, sample_size: int = 100) -> List:
    """Divide a list into smaller samples of a given size.

    Args:
        input_list (List): The input list to divide.
        sample_size (int, optional): The size of each sample. Defaults to 100.

    Returns:
        List: A list of divided samples.
    """

    divided_lists = []
    for i in range(0, len(input_list), sample_size):
        sample = input_list[i:i + sample_size]
        divided_lists.append(sample)
    return divided_lists



def GPT_automatic_translator(
        api_key: str = None,
        dataset: List = None,
        prefix_original: str = 'guanaco_set',
        prefix_translated: str = 'translated_set',
        sample_size: Optional[int]  = 100,
        sleep: Optional[int] = 120) -> None:
    """Translate a dataset to Spanish as default, but you can change to other language modifying the prompt,
     it handles TimeoutError and resuming from the point of failure if you have a low internet connexion or if there are  bad gateway errors.
     it saves the data into pkl format.

    Args:
        api_key (str): Your API key for the OpenAI API. Defaults to None.
        dataset (List): The dataset to translate. Defaults to None.
        sample_size (int, optional): The size of each translation sample. Defaults to 100.
        sleep (int, optional):The time between request to OpenAI API. Defaults to 120 seconds.
    Returns:
        None
    """
    if api_key == None:
        raise ValueError('You need and Open AI API key to perform the request')
    
    elif dataset == None:
        raise ValueError('You need a list of phrases to translate')
    # Load previously translated sets as the starting point
    loaded_sets = load_lists_from_files(prefix="translated_set")

    # Calculate the start index based on the number of loaded sets
    start_index = len(loaded_sets)

    # Divide the input dataset into smaller sets
    divided_dataset = divide_list_into_samples(dataset, sample_size=sample_size)

    i = start_index
    while i < len(divided_dataset):
        set_to_translate = divided_dataset[i]
        translated_set_i = []

        for text in set_to_translate:
            prompt = f'''Translate the phrase to Spanish. If you find an error code from a programming language, do not translate it.
            The phrase is going to be surrounded by triple Q:

                QQQ{text}QQQ

                Translation:
            '''
            try:
                translated_text = get_completion(prompt=prompt, api_key=api_key)
            except Exception as e:
                print(f"An error occurred: {e}. Restarting translation from index {i}.")
                x = print(f"An error occurred: {e}. Restarting translation from index {i}.")
                with open(f'translate_error.txt', 'w', encoding='utf-8') as f:
                    f.write(x)
                    f.close()
                time.sleep(sleep)  # Wait for a minute before retrying
                
                continue  # Retry the translation from the same point

            translated_set_i.append(translated_text)

        if len(translated_set_i) == len(set_to_translate):
            # Save the original guanaco set
            save_lists_to_files([set_to_translate], prefix=prefix_original, start_index=i)

            # Save the translated set after each successful translation
            save_lists_to_files([translated_set_i], prefix=prefix_translated, start_index=i)
        else:
            print("Skipping saving due to error.")

        i += 1  # Move to the next set

    print("Translation process completed.")


def create_dataset_from_pkl(
    prefix: list[str] = ['translated_set', 'translated_test'],
    pattern_list: list[str] = [r'La frase va a estar rodeada por triple Q:\n\n                QQQ', r'QQQ']
) -> DatasetDict:
    '''The function creates a HF dataset from pkl files.
    
    Args:
    prefix (list[str]): the prefix of the file you wish to load.
    pattern_list (list[str]): list of patterns to be removed from the text.
    
    Returns:
    A HF DatasetDict object.'''

    flatten_dict_test = ''
    flatten_dict_train = ''

    if not prefix:
        raise ValueError("Prefix list argument must be provided.")

    for i in prefix:
        loaded_translation = load_lists_from_files(prefix=i)
        flatten = [str(i) for a in loaded_translation for i in a]
        for pattern in pattern_list:
            flatten = [re.sub(pattern, '', i) for i in flatten]
        
        if i == 'translated_set':
            flatten_dict_train = Dataset.from_dict(mapping={'text': flatten}) 
        elif i == 'translated_test':
            flatten_dict_test = Dataset.from_dict(mapping={'text': flatten})

    dict_complete = {'train': flatten_dict_train, 'test': flatten_dict_test}
    complete = DatasetDict(dict_complete)
    return complete
