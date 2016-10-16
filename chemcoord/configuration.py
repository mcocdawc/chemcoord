import configparser
import os
import copy
import warnings
from warnings import warn

def _give_default_file_path():
    HOME = os.path.expanduser('~')
    filepath = os.path.join(HOME, '.chemcoordrc')
    return filepath

def provide_default_settings():
    settings = {}
    settings['show_warnings'] = {}
    settings['show_warnings']['valency'] = True
    # The Cartesian().get_bonds() method will use or not use a lookup.
    # Greatly increases performance if True, but could introduce bugs
    # if the Cartesian().xyz_frame is changed manually.
    settings['defaults'] = {}
    settings['defaults']['use_lookup_internally'] = True
    settings['defaults']['atomic_radius_data'] = 'atomic_radius_cc'
    settings['defaults']['viewer'] = 'gv.exe'
    # settings['viewer'] = 'avogadro'
    # settings['viewer'] = 'molden'
    # settings['viewer'] = 'jmol'
    return settings

def write_configuration_file(filepath=_give_default_file_path(),
                             overwrite=False):
    """Currently this function works only under UNIX."""
    config = configparser.ConfigParser()
    config.read_dict(settings)

    if os.path.isfile(filepath) and not overwrite:
        try:
            raise FileExistsError
        except NameError: # because of python2
            warn('File exists already and overwrite is False (default).')
    else:
        with open(filepath, 'w') as configfile:
            config.write(configfile)

        try:
            import ctypes.windll
            MAGIG_NUMBER_FOR_HIDING = 2
            ctypes.windll.kernel32.SetFileAttributesW(
                filepath, MAGIG_NUMBER_FOR_HIDING)
        except ImportError:
            # If not on windows submodule windll does not exist and
            # on UNIX '.' in front of a filename is already enough.
            pass


def read_configuration_file(filepath=_give_default_file_path()):
    """Read ~/.chemcoordrc and changes settings inplace"""
    config = configparser.ConfigParser()
    config.read(filepath)
    def get_correct_type(section, key, config):
        """ Gives e.g. the boolean True for the string 'True'"""
        def getstring(section, key, config):
            return config[section][key]
        def getinteger(section, key, config):
            return config[section].getint(key)
        def getboolean(section, key, config):
            return config[section].getboolean(key)
        def getfloat(section, key, config):
            return config[section].getfloat(key)
        special_actions = {} # Something different than a string is expected
        special_actions['show_warnings'] = {}
        special_actions['show_warnings']['valency'] = getboolean
        special_actions['defaults'] = {}
        special_actions['defaults']['use_lookup_internally'] = getboolean
        try:
            return special_actions[section][key](section, key, config)
        except KeyError:
            return getstring(section, key, config)

    for section in config.sections():
        for key in config[section]:
            settings[section][key] = get_correct_type(section, key, config)
    return settings

settings = provide_default_settings()
read_configuration_file()
