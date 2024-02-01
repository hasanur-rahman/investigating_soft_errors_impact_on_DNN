import random
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

super_label_map = {
    0: "Small Animals",
    1: "Small Animals",
    2: "Big Animals",
    3: "Big Animals",
    4: "Big Animals",
    5: "Small Animals",
    6: "Small Animals",
    7: "Aerial Objects",
    8: "Aerial Objects",
    9: "Big Animals",
    10: "Aerial Objects",
    11: "Aerial Objects",
    12: "Aerial Objects",
    13: "Aerial Objects",
    14: "Aerial Objects",
    15: "Aerial Objects",
    16: "Aerial Objects",
    17: "Aerial Objects",
    18: "Aerial Objects",
    19: "Aerial Objects",
    20: "Aerial Objects",
    21: "Aerial Objects",
    22: "Aerial Objects",
    23: "Aerial Objects",
    24: "Aerial Objects",
    25: "Small Animals",
    26: "Small Animals",
    27: "Small Animals",
    28: "Small Animals",
    29: "Small Animals",
    30: "Small Animals",
    31: "Small Animals",
    32: "Small Animals",
    33: "Small Animals",
    34: "Small Animals",
    35: "Small Animals",
    36: "Small Animals",
    37: "Small Animals",
    38: "Small Animals",
    39: "Small Animals",
    40: "Small Animals",
    41: "Small Animals",
    42: "Small Animals",
    43: "Small Animals",
    44: "Small Animals",
    45: "Small Animals",
    46: "Small Animals",
    47: "Small Animals",
    48: "Small Animals",
    49: "Small Animals",
    50: "Small Animals",
    51: "Big Animals",
    52: "Small Animals",
    53: "Small Animals",
    54: "Small Animals",
    55: "Small Animals",
    56: "Small Animals",
    57: "Small Animals",
    58: "Small Animals",
    59: "Small Animals",
    60: "Small Animals",
    61: "Small Animals",
    62: "Small Animals",
    63: "Small Animals",
    64: "Small Animals",
    65: "Small Animals",
    66: "Small Animals",
    67: "Small Animals",
    68: "Small Animals",
    69: "Small Animals",
    70: "Small Animals",
    71: "Small Animals",
    72: "Small Animals",
    73: "Small Animals",
    74: "Small Animals",
    75: "Small Animals",
    76: "Small Animals",
    77: "Small Animals",
    78: "Small Animals",
    79: "Small Animals",
    80: "Aerial Objects",
    81: "Aerial Objects",
    82: "Aerial Objects",
    83: "Aerial Objects",
    84: "Aerial Objects",
    85: "Aerial Objects",
    86: "Aerial Objects",
    87: "Aerial Objects",
    88: "Aerial Objects",
    89: "Aerial Objects",
    90: "Aerial Objects",
    91: "Aerial Objects",
    92: "Aerial Objects",
    93: "Aerial Objects",
    94: "Aerial Objects",
    95: "Aerial Objects",
    96: "Aerial Objects",
    97: "Aerial Objects",
    98: "Aerial Objects",
    99: "Aerial Objects",
    100: "Aerial Objects",
    101: "Big Animals",
    102: "Small Animals",
    103: "Big Animals",
    104: "Small Animals",
    105: "Small Animals",
    106: "Small Animals",
    107: "Small Animals",
    108: "Small Animals",
    109: "Small Animals",
    110: "Small Animals",
    111: "Small Animals",
    112: "Small Animals",
    113: "Small Animals",
    114: "Small Animals",
    115: "Small Animals",
    116: "Small Animals",
    117: "Small Animals",
    118: "Small Animals",
    119: "Small Animals",
    120: "Small Animals",
    121: "Small Animals",
    122: "Small Animals",
    123: "Small Animals",
    124: "Small Animals",
    125: "Small Animals",
    126: "Small Animals",
    127: "Aerial Objects",
    128: "Aerial Objects",
    129: "Aerial Objects",
    130: "Aerial Objects",
    131: "Aerial Objects",
    132: "Aerial Objects",
    133: "Aerial Objects",
    134: "Aerial Objects",
    135: "Aerial Objects",
    136: "Aerial Objects",
    137: "Aerial Objects",
    138: "Aerial Objects",
    139: "Aerial Objects",
    140: "Aerial Objects",
    141: "Aerial Objects",
    142: "Aerial Objects",
    143: "Aerial Objects",
    144: "Aerial Objects",
    145: "Aerial Objects",
    146: "Aerial Objects",
    147: "Big Animals",
    148: "Big Animals",
    149: "Big Animals",
    150: "Big Animals",
    151: "Big Animals",
    152: "Big Animals",
    153: "Big Animals",
    154: "Big Animals",
    155: "Big Animals",
    156: "Big Animals",
    157: "Big Animals",
    158: "Big Animals",
    159: "Big Animals",
    160: "Big Animals",
    161: "Big Animals",
    162: "Big Animals",
    163: "Big Animals",
    164: "Big Animals",
    165: "Big Animals",
    166: "Big Animals",
    167: "Big Animals",
    168: "Big Animals",
    169: "Big Animals",
    170: "Big Animals",
    171: "Big Animals",
    172: "Big Animals",
    173: "Big Animals",
    174: "Big Animals",
    175: "Big Animals",
    176: "Big Animals",
    177: "Big Animals",
    178: "Big Animals",
    179: "Big Animals",
    180: "Big Animals",
    181: "Big Animals",
    182: "Big Animals",
    183: "Big Animals",
    184: "Big Animals",
    185: "Big Animals",
    186: "Big Animals",
    187: "Big Animals",
    188: "Big Animals",
    189: "Big Animals",
    190: "Big Animals",
    191: "Big Animals",
    192: "Big Animals",
    193: "Big Animals",
    194: "Big Animals",
    195: "Big Animals",
    196: "Big Animals",
    197: "Big Animals",
    198: "Big Animals",
    199: "Big Animals",
    200: "Big Animals",
    201: "Big Animals",
    202: "Big Animals",
    203: "Big Animals",
    204: "Big Animals",
    205: "Big Animals",
    206: "Big Animals",
    207: "Big Animals",
    208: "Big Animals",
    209: "Big Animals",
    210: "Big Animals",
    211: "Big Animals",
    212: "Big Animals",
    213: "Big Animals",
    214: "Big Animals",
    215: "Big Animals",
    216: "Big Animals",
    217: "Big Animals",
    218: "Big Animals",
    219: "Big Animals",
    220: "Big Animals",
    221: "Big Animals",
    222: "Big Animals",
    223: "Big Animals",
    224: "Big Animals",
    225: "Big Animals",
    226: "Big Animals",
    227: "Big Animals",
    228: "Big Animals",
    229: "Big Animals",
    230: "Big Animals",
    231: "Big Animals",
    232: "Big Animals",
    233: "Big Animals",
    234: "Big Animals",
    235: "Big Animals",
    236: "Big Animals",
    237: "Big Animals",
    238: "Big Animals",
    239: "Big Animals",
    240: "Big Animals",
    241: "Big Animals",
    242: "Big Animals",
    243: "Big Animals",
    244: "Big Animals",
    245: "Big Animals",
    246: "Big Animals",
    247: "Big Animals",
    248: "Big Animals",
    249: "Big Animals",
    250: "Big Animals",
    251: "Big Animals",
    252: "Big Animals",
    253: "Big Animals",
    254: "Big Animals",
    255: "Big Animals",
    256: "Big Animals",
    257: "Big Animals",
    258: "Big Animals",
    259: "Big Animals",
    260: "Big Animals",
    261: "Big Animals",
    262: "Big Animals",
    263: "Big Animals",
    264: "Big Animals",
    265: "Big Animals",
    266: "Big Animals",
    267: "Big Animals",
    268: "Big Animals",
    269: "Big Animals",
    270: "Big Animals",
    271: "Big Animals",
    272: "Big Animals",
    273: "Big Animals",
    274: "Big Animals",
    275: "Big Animals",
    276: "Big Animals",
    277: "Big Animals",
    278: "Big Animals",
    279: "Big Animals",
    280: "Big Animals",
    281: "Big Animals",
    282: "Big Animals",
    283: "Big Animals",
    284: "Big Animals",
    285: "Big Animals",
    286: "Big Animals",
    287: "Big Animals",
    288: "Big Animals",
    289: "Big Animals",
    290: "Big Animals",
    291: "Big Animals",
    292: "Big Animals",
    293: "Big Animals",
    294: "Big Animals",
    295: "Big Animals",
    296: "Big Animals",
    297: "Big Animals",
    298: "Small Animals",
    299: "Small Animals",
    300: "Small Animals",
    301: "Small Animals",
    302: "Small Animals",
    303: "Small Animals",
    304: "Small Animals",
    305: "Small Animals",
    306: "Small Animals",
    307: "Small Animals",
    308: "Small Animals",
    309: "Small Animals",
    310: "Small Animals",
    311: "Small Animals",
    312: "Small Animals",
    313: "Small Animals",
    314: "Small Animals",
    315: "Small Animals",
    316: "Small Animals",
    317: "Small Animals",
    318: "Small Animals",
    319: "Small Animals",
    320: "Small Animals",
    321: "Small Animals",
    322: "Small Animals",
    323: "Small Animals",
    324: "Small Animals",
    325: "Small Animals",
    326: "Small Animals",
    327: "Small Animals",
    328: "Small Animals",
    329: "Small Animals",
    330: "Small Animals",
    331: "Small Animals",
    332: "Small Animals",
    333: "Small Animals",
    334: "Small Animals",
    335: "Small Animals",
    336: "Small Animals",
    337: "Small Animals",
    338: "Small Animals",
    339: "Big Animals",
    340: "Big Animals",
    341: "Big Animals",
    342: "Big Animals",
    343: "Big Animals",
    344: "Big Animals",
    345: "Big Animals",
    346: "Big Animals",
    347: "Big Animals",
    348: "Big Animals",
    349: "Big Animals",
    350: "Big Animals",
    351: "Big Animals",
    352: "Big Animals",
    353: "Big Animals",
    354: "Big Animals",
    355: "Big Animals",
    356: "Small Animals",
    357: "Small Animals",
    358: "Small Animals",
    359: "Small Animals",
    360: "Small Animals",
    361: "Small Animals",
    362: "Small Animals",
    363: "Small Animals",
    364: "Small Animals",
    365: "Small Animals",
    366: "Small Animals",
    367: "Small Animals",
    368: "Small Animals",
    369: "Small Animals",
    370: "Small Animals",
    371: "Small Animals",
    372: "Small Animals",
    373: "Small Animals",
    374: "Small Animals",
    375: "Small Animals",
    376: "Small Animals",
    377: "Small Animals",
    378: "Small Animals",
    379: "Small Animals",
    380: "Small Animals",
    381: "Small Animals",
    382: "Small Animals",
    383: "Small Animals",
    384: "Small Animals",
    385: "Big Animals",
    386: "Big Animals",
    387: "Big Animals",
    388: "Big Animals",
    389: "Small Animals",
    390: "Small Animals",
    391: "Small Animals",
    392: "Small Animals",
    393: "Small Animals",
    394: "Small Animals",
    395: "Small Animals",
    396: "Small Animals",
    397: "Small Animals",
    398: "Tools and Household Chores",
    399: "Tools and Household Chores",
    400: "Tools and Household Chores",
    401: "Tools and Household Chores",
    402: "Tools and Household Chores",
    403: "Water Vehicles",
    404: "Aerial Objects",
    405: "Aerial Objects",
    406: "Geological Formation and Structure",
    407: "Emergency Vehicle",
    408: "Four Wheeler",
    409: "Tools and Household Chores",
    410: "Geological Formation and Structure",
    411: "Tools and Household Chores",
    412: "Tools and Household Chores",
    413: "Tools and Household Chores",
    414: "Tools and Household Chores",
    415: "Geological Formation and Structure",
    416: "Tools and Household Chores",
    417: "Aerial Objects",
    418: "Tools and Household Chores",
    419: "Tools and Household Chores",
    420: "Tools and Household Chores",
    421: "Geological Formation and Structure",
    422: "Tools and Household Chores",
    423: "Tools and Household Chores",
    424: "Geological Formation and Structure",
    425: "Geological Formation and Structure",
    426: "Tools and Household Chores",
    427: "Tools and Household Chores",
    428: "Four Wheeler",
    429: "Tools and Household Chores",
    430: "Tools and Household Chores",
    431: "Tools and Household Chores",
    432: "Tools and Household Chores",
    433: "Tools and Household Chores",
    434: "Tools and Household Chores",
    435: "Tools and Household Chores",
    436: "Four Wheeler",
    437: "Geological Formation and Structure",
    438: "Tools and Household Chores",
    439: "Tools and Household Chores",
    440: "Tools and Household Chores",
    441: "Tools and Household Chores",
    442: "Tools and Household Chores",
    443: "Tools and Household Chores",
    444: "Two Wheeler",
    445: "Tools and Household Chores",
    446: "Tools and Household Chores",
    447: "Tools and Household Chores",
    448: "Tools and Household Chores",
    449: "Geological Formation and Structure",
    450: "Four Wheeler",
    451: "Tools and Household Chores",
    452: "Tools and Household Chores",
    453: "Tools and Household Chores",
    454: "Geological Formation and Structure",
    455: "Tools and Household Chores",
    456: "Tools and Household Chores",
    457: "Tools and Household Chores",
    458: "Geological Formation and Structure",
    459: "Tools and Household Chores",
    460: "Geological Formation and Structure",
    461: "Tools and Household Chores",
    462: "Tools and Household Chores",
    463: "Tools and Household Chores",
    464: "Tools and Household Chores",
    465: "Tools and Household Chores",
    466: "Four Wheeler",
    467: "Geological Formation and Structure",
    468: "Four Wheeler",
    469: "Tools and Household Chores",
    470: "Tools and Household Chores",
    471: "Tools and Household Chores",
    472: "Water Vehicles",
    473: "Tools and Household Chores",
    474: "Tools and Household Chores",
    475: "Tools and Household Chores",
    476: "Tools and Household Chores",
    477: "Tools and Household Chores",
    478: "Tools and Household Chores",
    479: "Tools and Household Chores",
    480: "Tools and Household Chores",
    481: "Tools and Household Chores",
    482: "Tools and Household Chores",
    483: "Geological Formation and Structure",
    484: "Water Vehicles",
    485: "Tools and Household Chores",
    486: "Tools and Household Chores",
    487: "Tools and Household Chores",
    488: "Tools and Household Chores",
    489: "Geological Formation and Structure",
    490: "Tools and Household Chores",
    491: "Tools and Household Chores",
    492: "Tools and Household Chores",
    493: "Tools and Household Chores",
    494: "Tools and Household Chores",
    495: "Tools and Household Chores",
    496: "Tools and Household Chores",
    497: "Geological Formation and Structure",
    498: "Geological Formation and Structure",
    499: "Tools and Household Chores",
    500: "Geological Formation and Structure",
    501: "Tools and Household Chores",
    502: "Tools and Household Chores",
    503: "Tools and Household Chores",
    504: "Tools and Household Chores",
    505: "Tools and Household Chores",
    506: "Geological Formation and Structure",
    507: "Tools and Household Chores",
    508: "Tools and Household Chores",
    509: "Geological Formation and Structure",
    510: "Water Vehicles",
    511: "Four Wheeler",
    512: "Tools and Household Chores",
    513: "Tools and Household Chores",
    514: "Tools and Household Chores",
    515: "Tools and Household Chores",
    516: "Tools and Household Chores",
    517: "Aerial Objects",
    518: "Tools and Household Chores",
    519: "Tools and Household Chores",
    520: "Tools and Household Chores",
    521: "Tools and Household Chores",
    522: "Tools and Household Chores",
    523: "Tools and Household Chores",
    524: "Tools and Household Chores",
    525: "Geological Formation and Structure",
    526: "Tools and Household Chores",
    527: "Tools and Household Chores",
    528: "Tools and Household Chores",
    529: "Tools and Household Chores",
    530: "Tools and Household Chores",
    531: "Tools and Household Chores",
    532: "Tools and Household Chores",
    533: "Tools and Household Chores",
    534: "Tools and Household Chores",
    535: "Tools and Household Chores",
    536: "Geological Formation and Structure",
    537: "Four Wheeler",
    538: "Tools and Household Chores",
    539: "Tools and Household Chores",
    540: "Tools and Household Chores",
    541: "Tools and Household Chores",
    542: "Tools and Household Chores",
    543: "Tools and Household Chores",
    544: "Tools and Household Chores",
    545: "Tools and Household Chores",
    546: "Tools and Household Chores",
    547: "Four Wheeler",
    548: "Tools and Household Chores",
    549: "Tools and Household Chores",
    550: "Tools and Household Chores",
    551: "Tools and Household Chores",
    552: "Tools and Household Chores",
    553: "Tools and Household Chores",
    554: "Water Vehicles",
    555: "Emergency Vehicle",
    556: "Tools and Household Chores",
    557: "Tools and Household Chores",
    558: "Tools and Household Chores",
    559: "Tools and Household Chores",
    560: "Tools and Household Chores",
    561: "Four Wheeler",
    562: "Geological Formation and Structure",
    563: "Tools and Household Chores",
    564: "Tools and Household Chores",
    565: "Four Wheeler",
    566: "Tools and Household Chores",
    567: "Tools and Household Chores",
    568: "Tools and Household Chores",
    569: "Four Wheeler",
    570: "Tools and Household Chores",
    571: "Tools and Household Chores",
    572: "Tools and Household Chores",
    573: "Four Wheeler",
    574: "Tools and Household Chores",
    575: "Four Wheeler",
    576: "Water Vehicles",
    577: "Tools and Household Chores",
    578: "Tools and Household Chores",
    579: "Tools and Household Chores",
    580: "Geological Formation and Structure",
    581: "Geological Formation and Structure",
    582: "Geological Formation and Structure",
    583: "Tools and Household Chores",
    584: "Tools and Household Chores",
    585: "Tools and Household Chores",
    586: "Four Wheeler",
    587: "Tools and Household Chores",
    588: "Tools and Household Chores",
    589: "Tools and Household Chores",
    590: "Tools and Household Chores",
    591: "Tools and Household Chores",
    592: "Tools and Household Chores",
    593: "Tools and Household Chores",
    594: "Tools and Household Chores",
    595: "Tools and Household Chores",
    596: "Tools and Household Chores",
    597: "Tools and Household Chores",
    598: "Geological Formation and Structure",
    599: "Geological Formation and Structure",
    600: "Tools and Household Chores",
    601: "Tools and Household Chores",
    602: "Tools and Household Chores",
    603: "Four Wheeler",
    604: "Tools and Household Chores",
    605: "Tools and Household Chores",
    606: "Tools and Household Chores",
    607: "Tools and Household Chores",
    608: "Tools and Household Chores",
    609: "Four Wheeler",
    610: "Tools and Household Chores",
    611: "Tools and Household Chores",
    612: "Four Wheeler",
    613: "Tools and Household Chores",
    614: "Tools and Household Chores",
    615: "Tools and Household Chores",
    616: "Tools and Household Chores",
    617: "Tools and Household Chores",
    618: "Tools and Household Chores",
    619: "Tools and Household Chores",
    620: "Tools and Household Chores",
    621: "Tools and Household Chores",
    622: "Tools and Household Chores",
    623: "Tools and Household Chores",
    624: "Geological Formation and Structure",
    625: "Water Vehicles",
    626: "Tools and Household Chores",
    627: "Four Wheeler",
    628: "Water Vehicles",
    629: "Tools and Household Chores",
    630: "Tools and Household Chores",
    631: "Tools and Household Chores",
    632: "Tools and Household Chores",
    633: "Tools and Household Chores",
    634: "Geological Formation and Structure",
    635: "Tools and Household Chores",
    636: "Tools and Household Chores",
    637: "Tools and Household Chores",
    638: "Tools and Household Chores",
    639: "Tools and Household Chores",
    640: "Tools and Household Chores",
    641: "Tools and Household Chores",
    642: "Tools and Household Chores",
    643: "Tools and Household Chores",
    644: "Tools and Household Chores",
    645: "Tools and Household Chores",
    646: "Tools and Household Chores",
    647: "Tools and Household Chores",
    648: "Tools and Household Chores",
    649: "Geological Formation and Structure",
    650: "Tools and Household Chores",
    651: "Tools and Household Chores",
    652: "Tools and Household Chores",
    653: "Tools and Household Chores",
    654: "Four Wheeler",
    655: "Tools and Household Chores",
    656: "Four Wheeler",
    657: "Tools and Household Chores",
    658: "Tools and Household Chores",
    659: "Tools and Household Chores",
    660: "Geological Formation and Structure",
    661: "Four Wheeler",
    662: "Tools and Household Chores",
    663: "Geological Formation and Structure",
    664: "Tools and Household Chores",
    665: "Two Wheeler",
    666: "Tools and Household Chores",
    667: "Tools and Household Chores",
    668: "Geological Formation and Structure",
    669: "Tools and Household Chores",
    670: "Two Wheeler",
    671: "Two Wheeler",
    672: "Geological Formation and Structure",
    673: "Tools and Household Chores",
    674: "Tools and Household Chores",
    675: "Four Wheeler",
    676: "Tools and Household Chores",
    677: "Tools and Household Chores",
    678: "Tools and Household Chores",
    679: "Tools and Household Chores",
    680: "Tools and Household Chores",
    681: "Tools and Household Chores",
    682: "Geological Formation and Structure",
    683: "Tools and Household Chores",
    684: "Tools and Household Chores",
    685: "Tools and Household Chores",
    686: "Tools and Household Chores",
    687: "Tools and Household Chores",
    688: "Tools and Household Chores",
    689: "Tools and Household Chores",
    690: "Four Wheeler",
    691: "Tools and Household Chores",
    692: "Tools and Household Chores",
    693: "Tools and Household Chores",
    694: "Tools and Household Chores",
    695: "Tools and Household Chores",
    696: "Tools and Household Chores",
    697: "Tools and Household Chores",
    698: "Geological Formation and Structure",
    699: "Tools and Household Chores",
    700: "Tools and Household Chores",
    701: "Tools and Household Chores",
    702: "Tools and Household Chores",
    703: "Tools and Household Chores",
    704: "Tools and Household Chores",
    705: "Four Wheeler",
    706: "Geological Formation and Structure",
    707: "Tools and Household Chores",
    708: "Geological Formation and Structure",
    709: "Tools and Household Chores",
    710: "Tools and Household Chores",
    711: "Tools and Household Chores",
    712: "Tools and Household Chores",
    713: "Tools and Household Chores",
    714: "Tools and Household Chores",
    715: "Tools and Household Chores",
    716: "Geological Formation and Structure",
    717: "Four Wheeler",
    718: "Tools and Household Chores",
    719: "Tools and Household Chores",
    720: "Tools and Household Chores",
    721: "Tools and Household Chores",
    722: "Tools and Household Chores",
    723: "Tools and Household Chores",
    724: "Water Vehicles",
    725: "Tools and Household Chores",
    726: "Tools and Household Chores",
    727: "Geological Formation and Structure",
    728: "Tools and Household Chores",
    729: "Tools and Household Chores",
    730: "Tools and Household Chores",
    731: "Tools and Household Chores",
    732: "Tools and Household Chores",
    733: "Tools and Household Chores",
    734: "Emergency Vehicle",
    735: "Tools and Household Chores",
    736: "Tools and Household Chores",
    737: "Tools and Household Chores",
    738: "Tools and Household Chores",
    739: "Tools and Household Chores",
    740: "Tools and Household Chores",
    741: "Tools and Household Chores",
    742: "Tools and Household Chores",
    743: "Geological Formation and Structure",
    744: "Tools and Household Chores",
    745: "Tools and Household Chores",
    746: "Tools and Household Chores",
    747: "Tools and Household Chores",
    748: "Tools and Household Chores",
    749: "Tools and Household Chores",
    750: "Tools and Household Chores",
    751: "Four Wheeler",
    752: "Tools and Household Chores",
    753: "Tools and Household Chores",
    754: "Tools and Household Chores",
    755: "Tools and Household Chores",
    756: "Tools and Household Chores",
    757: "Four Wheeler",
    758: "Tools and Household Chores",
    759: "Tools and Household Chores",
    760: "Tools and Household Chores",
    761: "Tools and Household Chores",
    762: "Geological Formation and Structure",
    763: "Tools and Household Chores",
    764: "Tools and Household Chores",
    765: "Tools and Household Chores",
    766: "Tools and Household Chores",
    767: "Tools and Household Chores",
    768: "Tools and Household Chores",
    769: "Tools and Household Chores",
    770: "Tools and Household Chores",
    771: "Tools and Household Chores",
    772: "Tools and Household Chores",
    773: "Tools and Household Chores",
    774: "Tools and Household Chores",
    775: "Tools and Household Chores",
    776: "Tools and Household Chores",
    777: "Tools and Household Chores",
    778: "Tools and Household Chores",
    779: "Four Wheeler",
    780: "Water Vehicles",
    781: "Tools and Household Chores",
    782: "Tools and Household Chores",
    783: "Tools and Household Chores",
    784: "Tools and Household Chores",
    785: "Tools and Household Chores",
    786: "Tools and Household Chores",
    787: "Tools and Household Chores",
    788: "Tools and Household Chores",
    789: "Tools and Household Chores",
    790: "Tools and Household Chores",
    791: "Tools and Household Chores",
    792: "Tools and Household Chores",
    793: "Tools and Household Chores",
    794: "Tools and Household Chores",
    795: "Tools and Household Chores",
    796: "Tools and Household Chores",
    797: "Tools and Household Chores",
    798: "Tools and Household Chores",
    799: "Geological Formation and Structure",
    800: "Tools and Household Chores",
    801: "Tools and Household Chores",
    802: "Four Wheeler",
    803: "Four Wheeler",
    804: "Tools and Household Chores",
    805: "Tools and Household Chores",
    806: "Tools and Household Chores",
    807: "Tools and Household Chores",
    808: "Tools and Household Chores",
    809: "Tools and Household Chores",
    810: "Tools and Household Chores",
    811: "Tools and Household Chores",
    812: "Aerial Objects",
    813: "Tools and Household Chores",
    814: "Water Vehicles",
    815: "Small Animals",
    816: "Tools and Household Chores",
    817: "Four Wheeler",
    818: "Tools and Household Chores",
    819: "Tools and Household Chores",
    820: "Four Wheeler",
    821: "Geological Formation and Structure",
    822: "Tools and Household Chores",
    823: "Tools and Household Chores",
    824: "Tools and Household Chores",
    825: "Geological Formation and Structure",
    826: "Tools and Household Chores",
    827: "Tools and Household Chores",
    828: "Tools and Household Chores",
    829: "Four Wheeler",
    830: "Emergency Vehicle",
    831: "Tools and Household Chores",
    832: "Geological Formation and Structure",
    833: "Water Vehicles",
    834: "Tools and Household Chores",
    835: "Tools and Household Chores",
    836: "Tools and Household Chores",
    837: "Tools and Household Chores",
    838: "Tools and Household Chores",
    839: "Geological Formation and Structure",
    840: "Tools and Household Chores",
    841: "Tools and Household Chores",
    842: "Tools and Household Chores",
    843: "Tools and Household Chores",
    844: "Tools and Household Chores",
    845: "Tools and Household Chores",
    846: "Tools and Household Chores",
    847: "Emergency Vehicle",
    848: "Tools and Household Chores",
    849: "Tools and Household Chores",
    850: "Tools and Household Chores",
    851: "Tools and Household Chores",
    852: "Tools and Household Chores",
    853: "Tools and Household Chores",
    854: "Tools and Household Chores",
    855: "Tools and Household Chores",
    856: "Tools and Household Chores",
    857: "Tools and Household Chores",
    858: "Tools and Household Chores",
    859: "Tools and Household Chores",
    860: "Geological Formation and Structure",
    861: "Tools and Household Chores",
    862: "Tools and Household Chores",
    863: "Geological Formation and Structure",
    864: "Four Wheeler",
    865: "Geological Formation and Structure",
    866: "Four Wheeler",
    867: "Four Wheeler",
    868: "Tools and Household Chores",
    869: "Tools and Household Chores",
    870: "Two Wheeler",
    871: "Water Vehicles",
    872: "Tools and Household Chores",
    873: "Geological Formation and Structure",
    874: "Four Wheeler",
    875: "Tools and Household Chores",
    876: "Tools and Household Chores",
    877: "Geological Formation and Structure",
    878: "Tools and Household Chores",
    879: "Tools and Household Chores",
    880: "Two Wheeler",
    881: "Tools and Household Chores",
    882: "Tools and Household Chores",
    883: "Tools and Household Chores",
    884: "Tools and Household Chores",
    885: "Tools and Household Chores",
    886: "Tools and Household Chores",
    887: "Tools and Household Chores",
    888: "Geological Formation and Structure",
    889: "Tools and Household Chores",
    890: "Tools and Household Chores",
    891: "Tools and Household Chores",
    892: "Tools and Household Chores",
    893: "Tools and Household Chores",
    894: "Tools and Household Chores",
    895: "Aerial Objects",
    896: "Tools and Household Chores",
    897: "Tools and Household Chores",
    898: "Tools and Household Chores",
    899: "Tools and Household Chores",
    900: "Tools and Household Chores",
    901: "Tools and Household Chores",
    902: "Tools and Household Chores",
    903: "Tools and Household Chores",
    904: "Tools and Household Chores",
    905: "Tools and Household Chores",
    906: "Tools and Household Chores",
    907: "Tools and Household Chores",
    908: "Tools and Household Chores",
    909: "Tools and Household Chores",
    910: "Tools and Household Chores",
    911: "Tools and Household Chores",
    912: "Geological Formation and Structure",
    913: "Water Vehicles",
    914: "Water Vehicles",
    915: "Geological Formation and Structure",
    916: "Tools and Household Chores",
    917: "Tools and Household Chores",
    918: "Tools and Household Chores",
    919: "Geological Formation and Structure",
    920: "Geological Formation and Structure",
    921: "Tools and Household Chores",
    922: "Tools and Household Chores",
    923: "Tools and Household Chores",
    924: "Tools and Household Chores",
    925: "Tools and Household Chores",
    926: "Tools and Household Chores",
    927: "Tools and Household Chores",
    928: "Tools and Household Chores",
    929: "Tools and Household Chores",
    930: "Tools and Household Chores",
    931: "Tools and Household Chores",
    932: "Tools and Household Chores",
    933: "Tools and Household Chores",
    934: "Tools and Household Chores",
    935: "Tools and Household Chores",
    936: "Tools and Household Chores",
    937: "Tools and Household Chores",
    938: "Tools and Household Chores",
    939: "Tools and Household Chores",
    940: "Tools and Household Chores",
    941: "Tools and Household Chores",
    942: "Tools and Household Chores",
    943: "Tools and Household Chores",
    944: "Tools and Household Chores",
    945: "Tools and Household Chores",
    946: "Tools and Household Chores",
    947: "Tools and Household Chores",
    948: "Tools and Household Chores",
    949: "Tools and Household Chores",
    950: "Tools and Household Chores",
    951: "Tools and Household Chores",
    952: "Tools and Household Chores",
    953: "Tools and Household Chores",
    954: "Tools and Household Chores",
    955: "Tools and Household Chores",
    956: "Tools and Household Chores",
    957: "Tools and Household Chores",
    958: "Tools and Household Chores",
    959: "Tools and Household Chores",
    960: "Tools and Household Chores",
    961: "Tools and Household Chores",
    962: "Tools and Household Chores",
    963: "Tools and Household Chores",
    964: "Tools and Household Chores",
    965: "Tools and Household Chores",
    966: "Tools and Household Chores",
    967: "Tools and Household Chores",
    968: "Tools and Household Chores",
    969: "Tools and Household Chores",
    970: "Geological Formation and Structure",
    971: "Tools and Household Chores",
    972: "Geological Formation and Structure",
    973: "Geological Formation and Structure",
    974: "Geological Formation and Structure",
    975: "Geological Formation and Structure",
    976: "Geological Formation and Structure",
    977: "Geological Formation and Structure",
    978: "Geological Formation and Structure",
    979: "Geological Formation and Structure",
    980: "Geological Formation and Structure",
    981: "Person",
    982: "Person",
    983: "Person",
    984: "Tools and Household Chores",
    985: "Tools and Household Chores",
    986: "Tools and Household Chores",
    987: "Tools and Household Chores",
    988: "Tools and Household Chores",
    989: "Tools and Household Chores",
    990: "Tools and Household Chores",
    991: "Small Animals",
    992: "Small Animals",
    993: "Small Animals",
    994: "Small Animals",
    995: "Small Animals",
    996: "Small Animals",
    997: "Small Animals",
    998: "Tools and Household Chores",
    999: "Tools and Household Chores",
}


def find_ddc(main_class, predicted_class):
    main_super_label = super_label_map[main_class]
    predicted_super_label = super_label_map[predicted_class]
    if main_super_label != predicted_super_label:
        return True
    else:
        return False


def find_avmis(main_class, predicted_class):
    type1_super_groups = ['Big Animals', 'Emergency Vehicle', 'Two Wheeler', 'Four Wheeler', 'Geological Formation and Structure', 'Person']
    type2_super_groups = ['Small Animals', 'Tools and Household Chores', 'Aerial Objects', 'Water Vehicles']
    main_super_label = super_label_map[main_class]
    predicted_super_label = super_label_map[predicted_class]

    if main_super_label != predicted_super_label:
        if main_super_label in type1_super_groups and predicted_super_label in type1_super_groups:
            return True
        elif main_super_label in type1_super_groups and predicted_super_label in type2_super_groups:
            return True
        else:
            return False
    else:
        return False


def get_statistics_without_fault_injection(original_label_list, predicted_label_list):
    correct_classification = 0
    misclassified_avmis = 0
    misclassified_non_avmis = 0
    correct_indices = []
    avmis_indexes = []
    non_avmis_indexes = []
    for i in range(len(original_label_list)):
        org_val = original_label_list[i]
        pred_val = predicted_label_list[i]
        if org_val != pred_val:
            if find_avmis(org_val, pred_val):
                misclassified_avmis += 1
                avmis_indexes.append(i)
            else:
                misclassified_non_avmis += 1
                non_avmis_indexes.append(i)
        else:
            correct_classification += 1
            correct_indices.append(i)
    return correct_indices, avmis_indexes, non_avmis_indexes, correct_classification, misclassified_avmis, misclassified_non_avmis


def get_statistics_per_image_with_fault_injection(previous_predicted_label_list, faulty_label_list):
    benign_count = 0
    faulty_avmis = 0
    faulty_non_avmis = 0
    for i in range(len(previous_predicted_label_list)):
        prev_pred_val = previous_predicted_label_list[i]
        faulty_val = faulty_label_list[i]
        if prev_pred_val != faulty_val:
            if find_avmis(prev_pred_val, faulty_val):
                faulty_avmis += 1
            else:
                faulty_non_avmis += 1
        else:
            benign_count += 1
    return benign_count, faulty_avmis, faulty_non_avmis


def get_string(data_list, index, text, model_count):
    s = text + '\t'
    for i in range(0, model_count):
        s += '{:.6f}'.format(data_list[i][index]) + '\t'
    return s + '\n'


def main():
    dataset_name = 'imagenet'
    # model_names = ['VGG16', 'VGG19', 'MobileNet', 'MobileNetV2', 'ResNet50', 'ResNet101', 'ResNet152', 'Xception', 'DenseNet121', 'DenseNet169', 'InceptionResNetV2', 'InceptionV3']
    # model_names = ['VGG16']
    model_name = 'Xception'
    model_list = []
    data_list = []
    total_layers = 133
    skipped_list = []
    total_scm = 0
    # for model_name in model_names:
    #     model_list.append(model_name)
    data = []
    for l in range(total_layers):
        if l in skipped_list:
            data.append(0)
            continue
        file1 = open(dataset_name + '_logs/' + dataset_name + '_final_log_' + model_name + '_' + str(l) + '_layer.txt', 'r')
        lines = file1.readlines()
        y_val = []
        predicted_label_list = []
        faulty_list = []
        for i in range(1000):
            line_parts = [x.strip() for x in lines[i].split(':')]
            y_val.append(int(line_parts[2]))
            predicted_label_list.append(int(line_parts[3]))
            faulty_list.append(int(line_parts[4]))
        benign_count, faulty_avmis, faulty_non_avmis = get_statistics_per_image_with_fault_injection(predicted_label_list, faulty_list)
        print("Layer " + str(l) + " : Non-SCM " + str(faulty_non_avmis) + ", SCM " + str(faulty_avmis))
        # data.append(benign_count)
        data.append(faulty_avmis)
        # data.append(faulty_non_avmis)
        # data_list.append(data)
        total_scm += faulty_avmis

    second_data_list = []
    for d in data:
        second_data_list.append((d * 100) / total_scm)

    plt.rcParams.update({'font.size': 14})
    plt.xlabel("Layer Number")
    plt.ylabel("SCM Probability")
    plt.bar(list(range(0, total_layers)), second_data_list)
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    plt.savefig(dataset_name + '_' + model_name + '_layer.png')

    # print(data_list)

        # fi_results = []
        # fi_taken = []
        # for i in range(10000):
        #     fi_taken.append(0)
        #     fi_results.append([])
        # while len(lines) > 0:
        #     line_parts = [x.strip() for x in lines[0].split(':')]
        #     index = int(line_parts[0])
        #     fi_results[index].append([int(line_parts[2]), int(line_parts[3])])
        #     lines.pop(0)
        #
        # prev_pred_list = []
        # faulty_list = []
        # for i in range(3000):
        #     while True:
        #         image_index = random.choice(correct_indices)
        #         if fi_taken[image_index] < 5:
        #             break
        #     fi_data = fi_results[image_index][fi_taken[image_index]]
        #     faulty_list.append(fi_data[1])
        #     prev_pred_list.append(fi_data[0])
        #     fi_taken[image_index] += 1
        #
        # benign_count, faulty_avmis, faulty_non_avmis = get_statistics_per_image_with_fault_injection(prev_pred_list,
        #                                                                                          faulty_list)
        # data.append(benign_count)
        # data.append(faulty_avmis)
        # data.append(faulty_non_avmis)
        #
        # prev_pred_list = []
        # faulty_list = []
        # for i in range(3000):
        #     while True:
        #         image_index = random.choice(avmis_indexes)
        #         if fi_taken[image_index] < 30:
        #             break
        #     fi_data = fi_results[image_index][fi_taken[image_index]]
        #     faulty_list.append(fi_data[1])
        #     prev_pred_list.append(fi_data[0])
        #     fi_taken[image_index] += 1
        #
        # benign_count, faulty_avmis, faulty_non_avmis = get_statistics_per_image_with_fault_injection(prev_pred_list,
        #                                                                                          faulty_list)
        #
        # data.append(benign_count)
        # data.append(faulty_avmis)
        # data.append(faulty_non_avmis)
        #
        # prev_pred_list = []
        # faulty_list = []
        # for i in range(3000):
        #     while True:
        #         image_index = random.choice(non_avmis_indexes)
        #         if fi_taken[image_index] < 30:
        #             break
        #     fi_data = fi_results[image_index][fi_taken[image_index]]
        #     faulty_list.append(fi_data[1])
        #     prev_pred_list.append(fi_data[0])
        #     fi_taken[image_index] += 1
        #
        # benign_count, faulty_avmis, faulty_non_avmis = get_statistics_per_image_with_fault_injection(prev_pred_list,
        #                                                                                          faulty_list)
        # data.append(benign_count)
        # data.append(faulty_avmis)
        # data.append(faulty_non_avmis)
        #
        # misclassified_indices = []
        # misclassified_indices.extend(avmis_indexes)
        # misclassified_indices.extend(non_avmis_indexes)
        # misclassified_indices.sort()
        # prev_pred_list = []
        # faulty_list = []
        # for i in range(3000):
        #     while True:
        #         image_index = random.choice(misclassified_indices)
        #         if fi_taken[image_index] < 30:
        #             break
        #     fi_data = fi_results[image_index][fi_taken[image_index]]
        #     faulty_list.append(fi_data[1])
        #     prev_pred_list.append(fi_data[0])
        #     fi_taken[image_index] += 1
        #
        # benign_count, faulty_avmis, faulty_non_avmis = get_statistics_per_image_with_fault_injection(prev_pred_list,
        #                                                                                              faulty_list)
        # data.append(benign_count)
        # data.append(faulty_avmis)
        # data.append(faulty_non_avmis)
        #
        # data_list.append(data)

    # print(model_list)
    # print(data_list)
    #
    # for i in range(0, len(model_list)):
    #     data_list[i][0] = data_list[i][0] / 100.0
    #     data_list[i][3] = data_list[i][3] / 30.0
    #     data_list[i][6] = data_list[i][6] / 30.0
    #     data_list[i][9] = data_list[i][9] / 30.0
    #     data_list[i][12] = data_list[i][12] / 30.0
    #     fault_free_total = (data_list[i][1] + data_list[i][2])
    #     data_list[i][1] = data_list[i][1] * 100 / fault_free_total
    #     data_list[i][2] = data_list[i][2] * 100 / fault_free_total
    #     initial_correct = (data_list[i][4] + data_list[i][5])
    #     data_list[i][4] = data_list[i][4] * 100 / initial_correct
    #     data_list[i][5] = data_list[i][5] * 100 / initial_correct
    #     initial_avmis = (data_list[i][7] + data_list[i][8])
    #     data_list[i][7] = data_list[i][7] * 100 / initial_avmis
    #     data_list[i][8] = data_list[i][8] * 100 / initial_avmis
    #     initial_non_avmis = (data_list[i][10] + data_list[i][11])
    #     data_list[i][10] = data_list[i][10] * 100 / initial_non_avmis
    #     data_list[i][11] = data_list[i][11] * 100 / initial_non_avmis
    #     initial_misclassified = (data_list[i][13] + data_list[i][14])
    #     data_list[i][13] = data_list[i][13] * 100 / initial_misclassified
    #     data_list[i][14] = data_list[i][14] * 100 / initial_misclassified
    #
    # file_object = open('graphs/' + dataset_name + '_new_results_golden_normalized.txt', 'a')
    # file_object.write(get_string(data_list, 2, 'Non-SCM', len(model_list)))
    # file_object.write(get_string(data_list, 1, 'SCM', len(model_list)))
    #
    # file_object = open('graphs/' + dataset_name + '_new_results_initial_correct_normalized.txt', 'a')
    # file_object.write(get_string(data_list, 5, 'Non-SCM', len(model_list)))
    # file_object.write(get_string(data_list, 4, 'SCM', len(model_list)))
    #
    # file_object = open('graphs/' + dataset_name + '_new_results_initial_avmis_normalized.txt', 'a')
    # file_object.write(get_string(data_list, 8, 'Non-SCM', len(model_list)))
    # file_object.write(get_string(data_list, 7, 'SCM', len(model_list)))
    #
    # file_object = open('graphs/' + dataset_name + '_new_results_initial_non_avmis_normalized.txt', 'a')
    # file_object.write(get_string(data_list, 11, 'Non-SCM', len(model_list)))
    # file_object.write(get_string(data_list, 10, 'SCM', len(model_list)))
    #
    # file_object = open('graphs/' + dataset_name + '_new_results_initial_misclassified_normalized.txt', 'a')
    # file_object.write(get_string(data_list, 14, 'Non-SCM', len(model_list)))
    # file_object.write(get_string(data_list, 13, 'SCM', len(model_list)))
    #
    # file_object = open('graphs/' + dataset_name + '_new_results_benign.txt', 'a')
    # file_object.write(get_string(data_list, 0, 'Accuracy', len(model_list)))
    # file_object.write(get_string(data_list, 3, 'Benign while correct', len(model_list)))
    # file_object.write(get_string(data_list, 6, 'Benign while avmis', len(model_list)))
    # file_object.write(get_string(data_list, 9, 'Benign while non avmis', len(model_list)))
    # file_object.write(get_string(data_list, 12, 'Benign while misclassified', len(model_list)))
    #
    # print(model_list)
    # print(data_list)


if __name__ == '__main__':
    main()
