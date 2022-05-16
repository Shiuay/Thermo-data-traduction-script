# -*- coding: utf-8 -*-
"""

Translates the species in thermo_data_file_1 to thermo_data_file_2.

"""

"""----------------------------------------------------------------------------
                                  IMPORTS
----------------------------------------------------------------------------"""

import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt

"""----------------------------------------------------------------------------
                                 PARAMETERS
----------------------------------------------------------------------------"""

N = 350 # Number of temperature points in Gibbs free energy
thermo_data_file_1 = "New_Curran_NOX_GLARBORG_therm.dat"
thermo_data_file_2 = "NASA.therm"

"""----------------------------------------------------------------------------
                   NASA COMPUTED THERMODYNAMIC FUNCTIONS
----------------------------------------------------------------------------"""

def cp(nasa_coeffs, temp):
    a = nasa_coeffs
    t = temp
    return a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4

def h(nasa_coeffs, temp):
    a = nasa_coeffs
    t = temp
    return a[0] + a[1]*t/2 + a[2]*t**2/3 + a[3]*t**3/4 + a[4]*t**4/5 + a[5]/t

def s(nasa_coeffs, temp):
    a = nasa_coeffs
    t = temp
    return a[0]*np.log(t) + a[1]*t + a[2]*t**2/2 + a[3]*t**3/3 + a[4]*t**4/4 + a[6]

"""----------------------------------------------------------------------------
                                    CLASSES
----------------------------------------------------------------------------"""

class ThermoData:
    def __init__(self, name=str(), atomic_composition=dict(), low_temp=float(), mid_temp=float(),
                 high_temp=float(), low_temp_nasa=list(), high_temp_nasa=list()):
        self.name = name
        self.atomic_composition = atomic_composition
        self.high_temp = high_temp
        self.mid_temp = mid_temp
        self.low_temp = low_temp
        self.low_temp_nasa = low_temp_nasa
        self.high_temp_nasa = high_temp_nasa

    def __repr__(self):
        return self.name

    def calc_therm(self, f):
        low_temps = np.linspace(self.low_temp, self.mid_temp, N)
        high_temps = np.linspace(self.mid_temp, self.high_temp, N)
        high_temp_cp = f(self.high_temp_nasa, high_temps)
        low_temp_cp = f(self.low_temp_nasa, low_temps)
        return np.hstack((low_temps, high_temps)), np.hstack((low_temp_cp, high_temp_cp))

    def new_calc_therm(self, temp, f):
        mid_temp_index = np.argmin(np.abs(temp - self.mid_temp))
        high_temps = temp[mid_temp_index:]
        low_temps = temp[:mid_temp_index]
        high_temp_cp = f(self.high_temp_nasa, high_temps)
        low_temp_cp = f(self.low_temp_nasa, low_temps)
        return np.hstack((low_temp_cp, high_temp_cp))

    def cp(self):
        return self.calc_therm(cp)

    def G(self):
        temp, a = self.calc_therm(s)
        temp, b = self.calc_therm(h)
        print(temp)
        return temp, a - b

    def G_new(self, temp):
        return self.new_calc_therm(temp, s) - self.new_calc_therm(temp, h)

class IsomerGroup:
    def __init__(self, atomic_composition=dict(), curran_species=list(),
                 pychegp_species=list()):
        self.atomic_composition = atomic_composition
        self.curran_species = curran_species
        self.pychegp_species = pychegp_species
        self.traductions = dict()

    def trad(self):
        curran_species = [therm.name for therm in self.curran_species]
        pychegp_species = [therm.name for therm in self.pychegp_species]
        trad_errors = np.zeros((len(curran_species), len(pychegp_species)))
        n_rows, n_columns = trad_errors.shape
        for i, curran_therm in enumerate(self.curran_species):
            for j, pychegp_therm in enumerate(self.pychegp_species):
                trad_errors[i][j] = calc_error(curran_therm, pychegp_therm)
        while 0 not in trad_errors.shape:
            i_min, j_min = np.unravel_index(trad_errors.argmin(), trad_errors.shape)
            if trad_errors[i_min][j_min] > 1:
                print("{} --> {} : {} %".format(curran_species[i_min],
                                              pychegp_species[j_min],
                                              trad_errors[i_min][j_min]))
            self.traductions[curran_species.pop(i_min)] = pychegp_species.pop(j_min)
            trad_errors = np.delete(trad_errors, i_min, 0)
            trad_errors = np.delete(trad_errors, j_min, 1)

class Traduction:
    def __init__(self, erreur=float(), species=dict()):
        self.erreur = erreur
        self.species =  species

"""----------------------------------------------------------------------------
                                 FUNCTIONS
----------------------------------------------------------------------------"""

def trim_and_uncomment(file_list):
    trimmed_file = list()
    thermo_found = False
    end_found = False
    for line in file_list:
        if "END" in line:
            end_found = True
        if thermo_found and not end_found:
            line = line.replace("\t", " "*4)
            comment_index = line.find("!")
            if comment_index != -1:
                decommented_line = line[:comment_index].strip("\n")
            else:
                decommented_line = line.strip("\n")
            trimmed_line = decommented_line.strip()
            if trimmed_line != "":
                trimmed_file.append(decommented_line)
        if "THERMO" in line:
            thermo_found = True
    return trimmed_file

def read_chemkin(file):
    try:
        tmin, tmoy, tmax = file[0].split()
    except:
        print("Default Temperature Intervals not Readable")

    is_species_line = False
    is_nasa_line = False
    was_species_line = False
    was_nasa_line = False
    therms = []
    therm = ThermoData()
    nasa_coeffs = []
    for index, line in enumerate(file[1:]):

        # Update info on line type
        was_species_line = is_species_line
        was_nasa_line = is_nasa_line
        if line[79] == "1":
            is_species_line = True
            is_nasa_line = False
        elif was_species_line or was_nasa_line:
            is_nasa_line = True
            is_species_line = False

        # Append nasa coeffs
        if was_nasa_line and not is_nasa_line:
            therm.high_temp_nasa = nasa_coeffs[:7]
            therm.low_temp_nasa = nasa_coeffs[7:]
            therms.append(therm)

        # Recover line data
        line = line[:75]
        if is_species_line:

            therm = ThermoData()
            nasa_coeffs = []

            # Recover species name
            therm.name = line[:24].split()[0]

            # Recover atomic composition
            atomic_composition = dict()
            for i in range(4):
                couple = line[24:44][i*5:(i+1)*5].split()
                if len(couple) == 2:
                    if int(float(couple[1])) != 0:
                        atomic_composition[couple[0].upper()] = int(float(couple[1]))
            therm.atomic_composition = atomic_composition

            # Recover high and low temperature intervals
            temps = [float(temp) for temp in line[45:75].split()]
            temps.sort()
            if len(temps) == 3:
                therm.high_temp = temps[-1]
                therm.mid_temp = temps[-2]
                therm.low_temp = temps[-3]
            else:
                print("WARNING : SPECIES {} HAVE {} TEMPERATURES".format(therm.name, len(temps)))

        elif is_nasa_line:

            # Append Nasa Coeffs
            for i in range(5):
                try:
                    nasa_coeffs.append(float(line[i*15:(i+1)*15]))
                except ValueError:
                    pass
            if len(file) - 2 == index:
                therm.high_temp_nasa = nasa_coeffs[:7]
                therm.low_temp_nasa = nasa_coeffs[7:]
                therms.append(therm)

    return therms

def read_pychegp(file):
    therms = []
    i = 0
    for line in file:
        line = line[:120]
        if line[0] == "#":
            continue
        else:
            line = line.split()
            if i % 5 == 0:
                therm = ThermoData()
                therm.name = line[0]
            elif i % 5 == 1:
                therm.atomic_composition = dict([(line[i*2].upper(), int(line[i*2+1]))
                for i in range(int(len(line)/2)) if int(line[i*2+1]) != 0])
            elif i % 5 == 2:
                therm.low_temp, therm.high_temp, therm.mid_temp = [float(temp) for temp in line]
            elif i % 5 == 3:
                therm.high_temp_nasa = [float(nasa_coeff) for nasa_coeff in line]
            elif i % 5 == 4:
                therm.low_temp_nasa = [float(nasa_coeff) for nasa_coeff in line]
                therms.append(therm)
            i += 1
    return therms

def calc_error(therm_curran, therm_pychegp):
    low_temp = max(therm_curran.low_temp, therm_pychegp.low_temp)
    high_temp = min(therm_curran.high_temp, therm_pychegp.high_temp)
    temp = np.linspace(low_temp, high_temp, N)
    G_curran = therm_curran.G_new(temp)
    G_pychegp = therm_pychegp.G_new(temp)
    norm = min(np.max(G_curran**2) - np.min(G_curran**2), np.max(G_pychegp**2) - np.min(G_pychegp**2))
    # norm = 1
    error = np.mean((G_curran-G_pychegp)**2)*100/norm
    return error

def get_isomer_group(isomer_groups, therm):
    for isomer_group in isomer_groups:
        if isomer_group.atomic_composition == therm.atomic_composition:
            return isomer_group

def get_atomic_number(atom, atomic_composition):
    try:
        return atomic_composition[atom]
    except KeyError:
        return 0

def comp_sort_key(atomic_composition):
    atoms_priority_list = ["N", "S", "C", "O", "H", "HE", "AR"]
    return [get_atomic_number(atom, atomic_composition) for atom in atoms_priority_list]

def dict_sort_key(atom):
    atoms_priority_list = ["N", "S", "C", "O", "H", "HE", "AR"]
    return atoms_priority_list.index(atom[0])

"""----------------------------------------------------------------------------
                                   SCRIPT
----------------------------------------------------------------------------"""

# Curran
with open(thermo_data_file_1, 'r') as file:
    curran_file = file.readlines()

with open(thermo_data_file_2, 'r') as file:
    pychegp_file = file.readlines()

therms_curran = read_chemkin(trim_and_uncomment(curran_file))
therms_pychegp = read_pychegp(pychegp_file)

atomic_compositions = []
for therm_curran in therms_curran:
    if therm_curran.atomic_composition not in atomic_compositions:
        atomic_compositions.append(therm_curran.atomic_composition)
for therm_pychegp in therms_pychegp:
    if therm_pychegp.atomic_composition not in atomic_compositions:
        atomic_compositions.append(therm_pychegp.atomic_composition)
atomic_compositions = [dict(sorted(atomic_composition.items(), key=dict_sort_key))
                       for atomic_composition in atomic_compositions]
atomic_compositions = sorted(atomic_compositions, key=comp_sort_key)
# print(atomic_compositions)

isomer_groups = [IsomerGroup(atomic_composition=atomic_composition, curran_species=list(), pychegp_species=list()) for atomic_composition in atomic_compositions]

for therm_curran in therms_curran:
    get_isomer_group(isomer_groups, therm_curran).curran_species.append(therm_curran)
for therm_pychegp in therms_pychegp:
    get_isomer_group(isomer_groups, therm_pychegp).pychegp_species.append(therm_pychegp)

for isomer_group in isomer_groups:
    # print(isomer_group.curran_species, isomer_group.pychegp_species)
    isomer_group.trad()
    print(isomer_group.traductions)
