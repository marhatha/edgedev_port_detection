import json

JSON_FILE = './dataset/homerouter/result.json'

def get_categories_from_json(json_file):
    """
    Extract category names from a COCO format JSON file.

    Parameters:
    - json_file: Path to the COCO format JSON file.

    Returns:
    - A list of category names.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    categories = {category['id']: category['name'] for category in data['categories']}
    # Sorting categories by ID to ensure they are in the correct order
    sorted_categories = [categories[cat_id] for cat_id in sorted(categories)]
    
    return sorted_categories

print(f"\nStep1\n")
categories = get_categories_from_json(JSON_FILE)
print(categories)

