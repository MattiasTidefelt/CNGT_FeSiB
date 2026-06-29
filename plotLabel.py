#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 10:44:21 2022

@author: ag0406
"""

# import re

# def plotLabel(string):
#     match1 = re.match(r"([a-zA-Z]+)([0-9]+)([a-zA-Z]+)([0-9]+)", string, re.I)
#     match2 = re.match(r"([a-zA-Z]+)([0-9]+)([a-zA-Z]+)", string, re.I)
#     match3 = re.match(r"([a-zA-Z]+)([0-9]+)", string, re.I)
#     match4 = re.match(r"([a-zA-Z]+)([0-9]+)([a-zA-Z]+)$", string, re.I)
#     match5 = re.match(r"(BCC|FCC|RHOMB_B)", string, re.I)

#     if match1:
#         items = match1.groups()
#         # Label = f'{items[0]}$_{items[1]}${items[2]}$_{items[3]}$'
#         Label = f'{items[0]}$_{{{items[1]}}}${items[2]}$_{{{items[3]}}}$'
#     elif match2:
#         items = match2.groups()
#         Label = f'{items[0]}$_{items[1]}${items[2]}'
#     elif match3:
#         items = match3.groups()
#         Label = f'{items[0]}$_{items[1]}$'
#     elif match4:
#         items = match4.groups()
#         Label = f'{items[0]}$_{items[1]}${items[2]}'
#     elif match5:
#         items = match5.groups()
#         Label = f'{items[0]}'
#     else:
#         Label = string  # Set a default label if no match is found

#     return Label

import re

def plotLabel(string):
    match0 = re.match(r"([a-zA-Z]+)([0-9]+)([a-zA-Z]+)([0-9]+)([a-zA-Z]+)([0-9]+)", string, re.I)
    match1 = re.match(r"([a-zA-Z]+)([0-9]+)([a-zA-Z]+)([0-9]+)", string, re.I)
    match2 = re.match(r"([a-zA-Z]+)([0-9]+)([a-zA-Z]+)", string, re.I)
    match3 = re.match(r"([a-zA-Z]+)([0-9]+)", string, re.I)
    match4 = re.match(r"([a-zA-Z]+)([0-9]+)([a-zA-Z]+)$", string, re.I)
    match5 = re.match(r"(BCC|FCC|RHOMB_B)", string, re.I)

    if match0:
        items = match0.groups()
        Label = f'{items[0]}$_{{{items[1]}}}${items[2]}$_{{{items[3]}}}${items[4]}$_{{{items[5]}}}$'
    elif match1:
        items = match1.groups()
        Label = f'{items[0]}$_{{{items[1]}}}${items[2]}$_{{{items[3]}}}$'
    elif match2:
        items = match2.groups()
        Label = f'{items[0]}$_{{{items[1]}}}${items[2]}'
    elif match3:
        items = match3.groups()
        Label = f'{items[0]}$_{{{items[1]}}}$'
    elif match4:
        items = match4.groups()
        Label = f'{items[0]}$_{{{items[1]}}}${items[2]}'
    elif match5:
        items = match5.groups()
        Label = f'{items[0]}'
    else:
        Label = string  # fallback

    return Label