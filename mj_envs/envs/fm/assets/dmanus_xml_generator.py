import os

from mj_envs.utils.xml_utils import parse_xml_with_comments, get_xml_str
import xml.etree.ElementTree as ET

'''
These functions are generally intended to reduce clutter in the xml
assets. The idea is to be able to generate xml files on the fly with 
specific configs, and not need to regenerate if there already exists an
xml file with the same config
'''

def generate_xml_config(params_xml, params_config):
    return generate_xml(params_xml), generate_config_file(params_config)

def generate_config_file(params):
    '''
    Generates .config files with sensor and actuators for mujoco Robot instances
    '''
    path_to_config = './'
    return path_to_config
    
def generate_xml(params):
    '''
    Generates xml files with requisite parameters
    '''
    path_to_xml = ''
    return path_to_xml