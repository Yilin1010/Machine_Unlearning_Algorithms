import json
# map class location to class index for imagenet label 


# map class locations to class indices without any other modifications and convertion
def map_clsidx_imagenet_collate_fn(data_info = 'data/tiny_infos.json',
                data_name = 'Maysee--tiny-imagenet'):

    synset_to_clsidx_map = get_synset_to_clsidx_map()

    # map clslocs to classids
    with open(data_info,'r') as f:
        dataset_info = json.load(f)
        tiny_imagenet_info = dataset_info[data_name]
        synsets = tiny_imagenet_info['features']['label']['names']

        clsloc_to_clsidx = [synset_to_clsidx_map.get(synset,-1) for synset in synsets]      
    

    def collate_fn(batch):
        
        # Map class locations to class indices
        for item in batch:
            item['label'] = clsloc_to_clsidx[item['label']]
        
        # Filter out items with invalid labels
        valid_batch = [item for item in batch if item['label'] != -1]
        
        return valid_batch

    return collate_fn
    

def get_synset_to_clsidx_map(file_path='data/synsets.txt'):
    synset_to_clsidx = {}
    with open(file_path, 'r') as file:
        for class_id, line in enumerate(file): # class id is index
            sysnet_id = line.strip() # only col is sysnet
            synset_to_clsidx[sysnet_id] = class_id
    return synset_to_clsidx


