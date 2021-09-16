from mj_envs.utils.xml_utils import parse_xml_with_comments, get_xml_str
import xml.etree.ElementTree as ET

def generate_xml_config(params_xml, params_config):
    return generate_xml(params_xml), generate_config_file(params_config)

def generate_config_file(params):
    '''
    Generates .config files with sensor and actuators for mujoco Robot instances
    '''
    config = []
    return config
    
def generate_xml(params):
    '''
    Generates xml files with requisite parameters
    '''
    xml = []
    return xml