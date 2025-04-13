import os
import pandas as pd
import ir_datasets
import requests

class CrisisFactsDataset:
    def __init__(self, event_list, save_folder="./dataset"):
        """
        Initializes the CrisisFactsDataset with a list of event numbers and a folder to save datasets.
        """
        self.event_list = event_list
        self.events_meta = {}
        self.dataset_keys = []
        self.save_folder = save_folder
        self.dataset_key_to_event = []
    
    def get_days_for_event(self, event_no):
        """
        Downloads and returns the list of days for a specific event.
        """
        url = f"http://trecis.org/CrisisFACTs/CrisisFACTS-{event_no}.requests.json"
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raises an error for bad responses
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving data for event {event_no}: {e}")
            return []
    
    def load_event_metadata(self):
        """
        Loads metadata for all events and constructs the dataset keys.
        """
        for event_no in self.event_list:
            daily_info = self.get_days_for_event(event_no)
            self.events_meta[event_no] = daily_info
            
            for day in daily_info:
                dataset_key = f"crisisfacts/{event_no}/{day['dateString']}"
                # print("  crisisfacts/"+event_no+"/"+day["dateString"], "-->", day["requestID"])
                dataset_key_translated = f"crisisfacts/{event_no}/{day['dateString'] + '_' + day['requestID']}"
                
                self.dataset_key_to_event.append(dataset_key_translated)
                self.dataset_keys.append(dataset_key)
        
        print(f"Loaded metadata for {len(self.event_list)} events.")
        return self.dataset_keys
    
    def get_dataset_event(self, event_no):
        """
        Returns the dataset for a specific event, loading it from disk if available or downloading it if not.
        """
        dataset = {}  # Store dataframes for each day
        days_event = [day['dateString'] for day in self.events_meta[event_no]]
        
        for day in days_event:
            file_path_data = os.path.join(self.save_folder, f"{event_no}_{day}_data.csv")
            # file_path_data_selected = f"./modified_dataset/{event_no}_{day}_data.csv"
            file_path_query = os.path.join(self.save_folder, f"{event_no}_{day}_query.csv")
            
            if os.path.exists(file_path_data) and os.path.exists(file_path_query):
                # Load from disk if files exist
                dataAsDF = pd.read_csv(file_path_data)
                queryAsDF = pd.read_csv(file_path_query)
                # dataAsDF_selected = pd.read_csv(file_path_data_selected)
            else:
                # Download the data if not saved yet
                dataset_obj = ir_datasets.load(f"crisisfacts/{event_no}/{day}")
                dataAsDF = pd.DataFrame(dataset_obj.docs_iter())  # Convert document iterator to DataFrame
                queryAsDF = pd.DataFrame(dataset_obj.queries_iter())  # Convert query iterator to DataFrame

                # Save to disk
                dataAsDF.to_csv(file_path_data, index=False)
                queryAsDF.to_csv(file_path_query, index=False)
            
            dataset[day] = {  # Save both data and queries as a dictionary entry
                "data": dataAsDF,
                "query": queryAsDF,
                # "data_modified": dataAsDF_selected
            }
        return dataset
    
    def get_dataset_event_no_connection(self, event_no):
        """
        Returns the dataset for a specific event, loading it from disk if available or downloading it if not.
        """
        dataset = {}  # Store dataframes for each day
        path = "./dataset"
        
        # file that starts with event_no
        files = [f for f in os.listdir(path) if f.startswith(str(event_no))]
        days_event = [f.split("_")[1] for f in files]
        
        for day in days_event:
            dataAsDF = pd.read_csv(os.path.join(path, f"{event_no}_{day}_data.csv"))
            queryAsDF = pd.read_csv(os.path.join(path, f"{event_no}_{day}_query.csv"))
            
            dataset[day] = {  # Save both data and queries as a dictionary entry
                "data": dataAsDF,
                "query": queryAsDF
            }
        
        return dataset
    