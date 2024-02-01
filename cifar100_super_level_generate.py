original_label_map = {
    0: 'apple',
    1: 'aquarium_fish',
    2: 'baby',
    3: 'bear',
    4: 'beaver',
    5: 'bed',
    6: 'bee',
    7: 'beetle',
    8: 'bicycle',
    9: 'bottle',
    10: 'bowl',
    11: 'boy',
    12: 'bridge',
    13: 'bus',
    14: 'butterfly',
    15: 'camel',
    16: 'can',
    17: 'castle',
    18: 'caterpillar',
    19: 'cattle',
    20: 'chair',
    21: 'chimpanzee',
    22: 'clock',
    23: 'cloud',
    24: 'cockroach',
    25: 'couch',
    26: 'crab',
    27: 'crocodile',
    28: 'cup',
    29: 'dinosaur',
    30: 'dolphin',
    31: 'elephant',
    32: 'flatfish',
    33: 'forest',
    34: 'fox',
    35: 'girl',
    36: 'hamster',
    37: 'house',
    38: 'kangaroo',
    39: 'keyboard',
    40: 'lamp',
    41: 'lawn_mower',
    42: 'leopard',
    43: 'lion',
    44: 'lizard',
    45: 'lobster',
    46: 'man',
    47: 'maple_tree',
    48: 'motorcycle',
    49: 'mountain',
    50: 'mouse',
    51: 'mushroom',
    52: 'oak_tree',
    53: 'orange',
    54: 'orchid',
    55: 'otter',
    56: 'palm_tree',
    57: 'pear',
    58: 'pickup_truck',
    59: 'pine_tree',
    60: 'plain',
    61: 'plate',
    62: 'poppy',
    63: 'porcupine',
    64: 'possum',
    65: 'rabbit',
    66: 'raccoon',
    67: 'ray',
    68: 'road',
    69: 'rocket',
    70: 'rose',
    71: 'sea',
    72: 'seal',
    73: 'shark',
    74: 'shrew',
    75: 'skunk',
    76: 'skyscraper',
    77: 'snail',
    78: 'snake',
    79: 'spider',
    80: 'squirrel',
    81: 'streetcar',
    82: 'sunflower',
    83: 'sweet_pepper',
    84: 'table',
    85: 'tank',
    86: 'telephone',
    87: 'television',
    88: 'tiger',
    89: 'tractor',
    90: 'train',
    91: 'trout',
    92: 'tulip',
    93: 'turtle',
    94: 'wardrobe',
    95: 'whale',
    96: 'willow_tree',
    97: 'wolf',
    98: 'woman',
    99: 'worm'
}

original_label_to_coarse_label = {
    'apple': 'fruit and vegetables',
    'aquarium_fish': 'fish',
    'baby': 'people',
    'bear': 'large carnivores',
    'beaver': 'aquatic mammals',
    'bed': 'household furniture',
    'bee': 'insects',
    'beetle': 'insects',
    'bicycle': 'Two wheelers',
    'bottle': 'food containers',
    'bowl': 'food containers',
    'boy': 'people',
    'bridge': 'large man-made outdoor things',
    'bus': 'Four wheelers',
    'butterfly': 'insects',
    'camel': 'large omnivores and herbivores',
    'can': 'food containers',
    'castle': 'large man-made outdoor things',
    'caterpillar': 'insects',
    'cattle': 'large omnivores and herbivores',
    'chair': 'household furniture',
    'chimpanzee': 'large omnivores and herbivores',
    'clock': 'household electrical devices',
    'cloud': 'large natural outdoor scenes',
    'cockroach': 'insects',
    'couch': 'household furniture',
    'crab': 'non-insect invertebrates',
    'crocodile': 'reptiles',
    'cup': 'food containers',
    'dinosaur': 'reptiles',
    'dolphin': 'aquatic mammals',
    'elephant': 'large omnivores and herbivores',
    'flatfish': 'fish',
    'forest': 'large natural outdoor scenes',
    'fox': 'medium-sized mammals',
    'girl': 'people',
    'hamster': 'small mammals',
    'house': 'large man-made outdoor things',
    'kangaroo': 'large omnivores and herbivores',
    'keyboard': 'household electrical devices',
    'lamp': 'household electrical devices',
    'lawn_mower': 'Four wheelers',
    'leopard': 'large carnivores',
    'lion': 'large carnivores',
    'lizard': 'reptiles',
    'lobster': 'non-insect invertebrates',
    'man': 'people',
    'maple_tree': 'trees',
    'motorcycle': 'Two wheelers',
    'mountain': 'large natural outdoor scenes',
    'mouse': 'small mammals',
    'mushroom': 'fruit and vegetables',
    'oak_tree': 'trees',
    'orange': 'fruit and vegetables',
    'orchid': 'flowers',
    'otter': 'aquatic mammals',
    'palm_tree': 'trees',
    'pear': 'fruit and vegetables',
    'pickup_truck': 'Four wheelers',
    'pine_tree': 'trees',
    'plain': 'large natural outdoor scenes',
    'plate': 'food containers',
    'poppy': 'flowers',
    'porcupine': 'medium-sized mammals',
    'possum': 'medium-sized mammals',
    'rabbit': 'small mammals',
    'raccoon': 'medium-sized mammals',
    'ray': 'fish',
    'road': 'large man-made outdoor things',
    'rocket': 'Four wheelers',
    'rose': 'flowers',
    'sea': 'large natural outdoor scenes',
    'seal': 'aquatic mammals',
    'shark': 'fish',
    'shrew': 'small mammals',
    'skunk': 'medium-sized mammals',
    'skyscraper': 'large man-made outdoor things',
    'snail': 'non-insect invertebrates',
    'snake': 'reptiles',
    'spider': 'non-insect invertebrates',
    'squirrel': 'small mammals',
    'streetcar': 'Four wheelers',
    'sunflower': 'flowers',
    'sweet_pepper': 'fruit and vegetables',
    'table': 'household furniture',
    'tank': 'Four wheelers',
    'telephone': 'household electrical devices',
    'television': 'household electrical devices',
    'tiger': 'large carnivores',
    'tractor': 'Four wheelers',
    'train': 'Four wheelers',
    'trout': 'fish',
    'tulip': 'flowers',
    'turtle': 'reptiles',
    'wardrobe': 'household furniture',
    'whale': 'aquatic mammals',
    'willow_tree': 'trees',
    'wolf': 'large carnivores',
    'woman': 'people',
    'worm': 'non-insect invertebrates'
}

# coarse_label_to_first_super_label = {
#     'fruit and vegetables': 'Flowers, fruits and vegetables',
#     'fish': 'Aquatic animals',
#     'people': 'people',
#     'large carnivores': 'Large animals',
#     'aquatic mammals': 'Aquatic animals',
#     'household furniture': 'Household furniture',
#     'insects': 'Insects and small animals',
#     'Two wheelers': 'Two wheelers',
#     'Four wheelers': 'Four wheelers',
#     'food containers': 'Household chores',
#     'household electrical devices': 'Household chores',
#     'reptiles': 'Aquatic animals',
#     'large man-made outdoor things': 'Large outdoor objects',
#     'trees': 'trees',
#     'large natural outdoor scenes': 'Large outdoor objects',
#     'flowers': 'Flowers, fruits and vegetables',
#     'non-insect invertebrates': 'Insects and small animals',
#     'small mammals': 'Insects and small animals',
#     'medium-sized mammals': 'Large animals',
#     'large omnivores and herbivores': 'Large animals'
# }
#
# first_super_label_to_second_super_label = {
#     'Flowers, fruits and vegetables': 'Small Objects',
#     'Aquatic animals': 'Large Objects',
#     'people': 'People',
#     'Large animals': 'Large Objects',
#     'Household furniture': 'Large Objects',
#     'Insects and small animals': 'Small Objects',
#     'Two wheelers': 'Two Wheelers',
#     'Four wheelers': 'Four Wheelers',
#     'Household chores': 'Small Objects',
#     'Large outdoor objects': 'Large outdoor objects',
#     'trees': 'Large Objects'
# }


coarse_label_to_first_super_label = {
    'fruit and vegetables': 'Household Objects',
    'fish': 'Small Animals',
    'people': 'People',
    'large carnivores': 'Large Animals',
    'aquatic mammals': 'Large Animals',
    'household furniture': 'Household Objects',
    'insects': 'Small Animals',
    'Two wheelers': 'Two Wheelers',
    'Four wheelers': 'Four Wheelers',
    'food containers': 'Household Objects',
    'household electrical devices': 'Household Objects',
    'reptiles': 'Small Animals',
    'large man-made outdoor things': 'Large Outdoor Objects',
    'trees': 'Trees',
    'large natural outdoor scenes': 'Large Outdoor Objects',
    'flowers': 'Household Objects',
    'non-insect invertebrates': 'Small Animals',
    'small mammals': 'Small Animals',
    'medium-sized mammals': 'Small Animals',
    'large omnivores and herbivores': 'Large Animals'
}

def get_super_label(org_label):
    return coarse_label_to_first_super_label[original_label_to_coarse_label[org_label]]


first_super_label_mappings = {}
for key in original_label_map:
    first_super_label_mappings[key] = get_super_label(original_label_map[key])

file_object = open('cifar100_new_super_label_mappings.txt', 'a')
file_object.write("{\n")
for key in first_super_label_mappings:
    file_object.write(str(key))
    file_object.write(':"')
    file_object.write(first_super_label_mappings[key].rstrip("\n"))
    file_object.write('",\n')
file_object.write("}\n")