import datajoint as dj

dj.config["enable_python_native_blobs"] = True
import numpy as np
import os
from datetime import datetime
import datetime
import h5py
from scipy import signal
import math
from configparser import ConfigParser
import pandas as pd
import pickle
from copy import deepcopy
from scipy import signal, cluster, stats
from numpy.linalg import lstsq
# import GPy
# import pycircstat
# from rfest import splineLG, splineLNP, splineLNLN
# from rfest import build_design_matrix, get_spatial_and_temporal_filters
from multiprocessing import Pool
import copy

schema = dj.schema('ageuler_amacrine_color', locals())


@schema
class UserInfo(dj.Manual):
    definition = """
    # Info for decoding file names

    experimenter                :varchar(255)             # path to header file, used for computed tables
    ---
    data_dir                    :varchar(255)             # path to header file, used for computed tables
    datatype_loc                :tinyint                  # string location for datatype (eg. imaging: SMP_)
    field_loc                   :tinyint                  # string location for field
    region_loc                  :tinyint                  # string location for region (eg. dorsal-temporal recording)
    stimulus_loc                :tinyint                  # string location for stimulus
    pharm_loc                   :tinyint                  # string location for pharmacology
    """

    def upload_user(self, userdict=None):
        uploaded = UserInfo().fetch("experimenter")
        for item in userdict:
            if item["experimenter"] not in uploaded:
                UserInfo().insert([item])  # more than one user can be uploaded at a time
            else:
                print("Information for that user already uploaded")


@schema
class Experiment(dj.Computed):
    definition = """
    # Header-File location, name and path
    -> UserInfo
    date                        :date                     # date of recording
    exp_num                     :mediumint                # experiment number in a day
    ---
    headername                  :varchar(255)             # path to header file, used for computed tables
    headerpath                  :varchar(255)             # path to header file, used for computed tables
    """

    def _make_tuples(self, key):

        data_directory = (UserInfo & key).fetch("data_dir")[0]
        self.__add_experiments(key, data_directory)

    def rescan_filesystem(self, *restrictions):

        user = (UserInfo & restrictions)
        experimenter = user.fetch("experimenter")[0]
        data_directory = user.fetch("data_dir")[0]

        key = {}
        key['experimenter'] = experimenter

        self.__add_experiments(key, data_directory, only_new=True, restrictions=restrictions)

    def __add_experiments(self, key, data_directory, only_new=False, restrictions="none"):

        # print(data_directory)
        # For data_directory in data_directories:
        # -- walk through all folders; select those containing header files
        os_walk_output = []
        for folder, subfolders, files in os.walk(data_directory):
            for file_ in files:
                if os.path.splitext(file_)[1] == '.ini':
                    os_walk_output.append(folder)

        for headerpath in os_walk_output:

            files = os.listdir(headerpath)
            headername = [s for s in files if ".ini" in s][0]

            date = headerpath[:headerpath.rfind("/")]
            date = date.split("/")[6]
            date = datetime.strptime(date, '%Y%m%d')

            exp_num = int(headerpath[headerpath.rfind("/") + 1:])

            # check if make all tupels or only scan for new files
            if only_new:
                # search for already added experiments, if found skip this
                search = (self & restrictions & 'date = "{}" and exp_num = "{}"'.format(date, exp_num))
                if len(search) > 0:
                    continue

            key["headerpath"] = headerpath + "/"
            key["headername"] = headername
            key["date"] = date
            key["exp_num"] = exp_num

            config_dict = {}
            parser = ConfigParser()
            parser.read(headerpath + "/" + headername)
            for key1 in parser.keys():
                for key2 in parser[key1].keys():
                    config_dict[key2[key2.find("_") + 1:]] = str(parser[key1][key2])

            self.insert1(key, allow_direct_insert=True)
            print(key['experimenter'], date)
            # Populate ExpInfo table for this experiment
            expinfo_key = {}
            expinfo_key['experimenter'] = key['experimenter']
            expinfo_key['date'] = date
            expinfo_key['exp_num'] = exp_num
            expinfo_key["eye"] = config_dict["eye"]
            expinfo_key["projname"] = config_dict["projname"]
            expinfo_key["setupid"] = config_dict["setupid"]
            expinfo_key["prep"] = config_dict["prep"]
            expinfo_key["preprem"] = config_dict["preprem"]
            expinfo_key["darkadapt_hrs"] = config_dict["darkadapt_hrs"]
            expinfo_key["slicethickness_um"] = config_dict["slicethickness_um"]
            expinfo_key["bathtemp_degc"] = config_dict["bathtemp_degc"]

            optdiscorien = config_dict["prepwmorient"]
            if optdiscorien is '':
                optdiscorien = -1
            expinfo_key["prepwmorient"] = optdiscorien
            # find out whether experimenter used brackets around the string and possibly remove them
            odpos_string = config_dict["prepwmopticdiscpos"]
            if odpos_string.find('(') >= 0:
                odpos_string = odpos_string.strip('()')
            # find out whether experimenter used "," or ";" as string separator
            separator = ";"
            if config_dict["prepwmopticdiscpos"].find(separator) == -1:
                separator = ','
            odpos_list = odpos_string.split(separator)
            if len(odpos_list) == 3:
                expinfo_key["odx"] = float(odpos_list[0])
                expinfo_key["ody"] = float(odpos_list[1])
                try:
                    expinfo_key["odz"] = float(odpos_list[2])
                except(ValueError):
                    expinfo_key["odz"] = 0
                expinfo_key["od_valid_flag"] = 1
            else:
                expinfo_key["odx"] = 0
                expinfo_key["ody"] = 0
                expinfo_key["odz"] = 0
                expinfo_key["od_valid_flag"] = 0

            # print(expinfo_key)
            ExpInfo().insert1(expinfo_key)

            # Populate Animal table for this experiment
            animal_key = {}
            animal_key['experimenter'] = key['experimenter']
            animal_key['date'] = date
            animal_key['exp_num'] = exp_num

            animal_key["genline"] = config_dict["genline"]
            animal_key["genbkglinerem"] = config_dict["genbkglinerem"]
            animal_key["genline_reporter"] = config_dict["genline_reporter"]
            animal_key["genline_reporterrem"] = config_dict["genline_reporterrem"]

            animal_species = config_dict["animspecies"]
            if animal_species == "":
                animal_species = "mouse"
            animal_key["animspecies"] = animal_species

            animal_key["animgender"] = config_dict["animgender"]
            animal_key["animdob"] = config_dict["animdob"]
            animal_key["animrem"] = config_dict["animrem"]

            # print(animal_key)
            Animal().insert1(animal_key)

            # Populate Indicator table for this experiment
            indicator_key = {}
            indicator_key['experimenter'] = key['experimenter']
            indicator_key['date'] = date
            indicator_key['exp_num'] = exp_num

            indicator_key["isepored"] = config_dict["isepored"]
            indicator_key["eporrem"] = config_dict["eporrem"]
            indicator_key["epordye"] = config_dict["epordye"]
            indicator_key["isvirusinject"] = config_dict["isvirusinject"]
            indicator_key["virusvect"] = config_dict["virusvect"]
            indicator_key["virusserotype"] = config_dict["virusserotype"]
            indicator_key["virustransprotein"] = config_dict["virustransprotein"]
            indicator_key["virusinjectq"] = config_dict["virusinjectq"]
            indicator_key["virusinjectrem"] = config_dict["virusinjectrem"]
            indicator_key["tracer"] = config_dict["tracer"]
            indicator_key["isbraininject"] = config_dict["isbraininject"]
            indicator_key["braininjectrem"] = config_dict["braininjectrem"]
            indicator_key["braininjectq"] = config_dict["braininjectq"]

            # print(indicator_key)
            Indicator().insert1(indicator_key)

            # Populate Pharmacology table for this experiment
            pharminfo_key = {}
            drug = config_dict.get("pharmdrug", "")
            if drug == "none" or len(drug) == 0:
                pharmaflag = 0
            else:
                pharmaflag = 1

            pharminfo_key['experimenter'] = key['experimenter']
            pharminfo_key['date'] = date
            pharminfo_key['exp_num'] = exp_num
            # pharminfo_key['field_id'] = (Field() & key).fetch("field_id")[0]

            pharminfo_key["pharmaflag"] = pharmaflag
            pharminfo_key['drug'] = drug
            pharminfo_key['pharmconc'] = config_dict.get("pharmdrugconc_um", "")
            # pharminfo_key['preapptime'] = config_dict.get("pretime", "")
            pharminfo_key['pharmcom'] = config_dict.get("pharmrem", "")

            PharmInfo().insert1(pharminfo_key)


@schema
class ExpInfo(dj.Manual):
    definition = """
    # General preparation details
    -> Experiment
    ---
    eye                         :enum("left", "right", "unknown") # left or right eye of the animal
    projname                    :varchar(255)                   # name of experimental project
    setupid                     :varchar(255)                   # setup 1-3
    prep="wholemount"           :enum("wholemount", "slice")    # preparation type of the retina
    preprem                     :varchar(255)                   # comments on the preparation
    darkadapt_hrs               :varchar(255)                   # time spent dark adapting animal before disection
    slicethickness_um           :varchar(255)                   # thickness of each slice in slice preparation
    bathtemp_degc               :varchar(255)                   # temperature of bath chamber
    prepwmorient                :smallint                       # retina orientation in chamber (0° = dorsal away from experimenter)
    odx                         :float                          # x location of optic disc as read in from .ini file (if available)
    ody                         :float                          # y location of optic disc as read in from .ini file (if available)
    odz                         :float                          # z location of optic disc as read in from .ini file (if available)
    od_valid_flag               :tinyint                        # flag (0, 1) indicating whether (1) or whether not (0)
                                                                # the optic disk position was documented in .ini file and
                                                                # is valid to use
    """


@schema
class Animal(dj.Manual):
    definition = """
    # Animal info and genetic background
    -> Experiment
    ---
    genline                   :varchar(255)                     # Genetic background line
    genbkglinerem             :varchar(255)                     # Comments about background line
    genline_reporter          :varchar(255)                     # Genetic reporter line
    genline_reporterrem       :varchar(255)                     # Comments about reporter line
    animspecies="mouse"       :enum("mouse","rat","zebrafish")  # animal species
    animgender                :varchar(255)                     # gender.
    animdob                   :varchar(255)                     # Whether to have this or DOB?
    animrem                   :varchar(255)                     # Comments about animal
    """


@schema
class Indicator(dj.Manual):
    definition = """
    # Indicator used for imaging
    -> Experiment
    ---
    isepored                    :varchar(255)                   # whether the retina was electroporated
    eporrem                     :varchar(255)                   # comments about the electroporation
    epordye                     :varchar(255)                   # which dye was used for the electroporation
    isvirusinject               :varchar(5)                     # whether the retina was injected
    virusvect                   :varchar(255)                   # what vector was used in the injection
    virusserotype               :varchar(255)                   # what serotype was used in the injection
    virustransprotein           :varchar(255)                   # the viral transprotein
    virusinjectrem              :varchar(255)                   # comments about the injection
    virusinjectq                :varchar(255)                   # numerical rating of the injection quality
    isbraininject               :varchar(5)                     # whether the retina was injected
    tracer                      :varchar(255)                   # which tracer has been used in the brain injection
    braininjectq                :varchar(255)                   # numerical rating of the brain injection quality
    braininjectrem              :varchar(255)                   # comments on the brain injection
    """


@schema
class PharmInfo(dj.Manual):
    definition = """
    # Pharmacology Info
    ->Experiment
    ---
    pharmaflag      :tinyint        # 1 there was pharma, 0 no pharma
    drug            :varchar(255)   # which drug was applied. Multiple drugs separated by ';'
    pharmconc       :varchar(255)   # concentration used in micromolar. Multple drugs, separate conc by ';'
    pharmcom        :varchar(255)   # experimenter comments
    """


@schema
class Field(dj.Computed):
    definition = """
    # Recording fields
    -> Experiment
    field_id                   :mediumint                      # automatic surrogate id; auto_increment?
    ---
    field                      :varchar(255)                   # string identifying files corresponding to field
    odflag                     :tinyint                        # flag wheter field is the optic disk
    locflag                    :tinyint                        # flag whether the field was recorded for its location eg edges of the retina
    recording_depth=-9999      :float                          # XY-scan: single element list with IPL depth, XZ: list of ROI depths (normalized to Chat bands, see paper)
    """

    class Zstack(dj.Part):
        definition = """
        #was there a zstack in the field
        ->Field
        ---
        zstack      :tinyint    #flag marking whether field was a zstack
        zstep       :float      #size of step size in um
        """

    class RoiMask(dj.Part):
        definition = """
        # ROI Mask
        -> Field
        ---
        fromfile                   :varchar(255)            # from which file the roi mask was extracted from
        roi_mask                   :longblob                # roi mask for the recording field
        absx                       :float                   # absolute position in the x axis as recorded by ScanM
        absy                       :float                   # absolute position in the y axis as recorded by ScanM
        absz                       :float                   # absolute position in the z axis as recorded by ScanM
        """

    def _make_tuples(self, key):
        self.__add_experiments(key)

    def rescan_filesystem(self, experimenter, date, exp_num):

        restrictions = dict(experimenter=experimenter,
                            date=date,
                            exp_num=exp_num)
        field_loc = (UserInfo() * Experiment() & restrictions).fetch1('field_loc')
        known_fields = (self.RoiMask() & restrictions).fetch("fromfile")
        known_field_names = [f.split("_")[field_loc] for f in known_fields]
        headerpath = (Experiment() & restrictions).fetch1('headerpath')
        data_directory = headerpath + 'Pre/'
        for _, _, files in os.walk(data_directory):
            all_fields = [f.split('_')[field_loc] for f in files]
        all_fields = np.unique(all_fields)
        missing_fields = [f for f in all_fields if not f in known_field_names]
        if len(missing_fields) > 0:
            # restrictions.pop("exp_num")
            self.__add_experiments(restrictions, known_fields=tuple(known_field_names))

    def __add_experiments(self, key, known_fields=()):

        headerpath = (Experiment() & key).fetch1('headerpath')
        field_loc = (UserInfo() * Experiment() & key).fetch1('field_loc')
        stimulus_loc = (UserInfo() * Experiment() & key).fetch1('stimulus_loc')
        region_loc = (UserInfo() * Experiment() & key).fetch1('region_loc')

        fields, stimuli, locflag, odflag = [], [], [], []

        # check if path is still alive, else you get an error at the bottom
        if os.path.exists(headerpath + "Pre"):

            print("processing Field in " + headerpath + "Pre ...")

            # create list with all filenames in "Pre" belonging to this experiment
            list_files = os.listdir(headerpath + "Pre")
            files = []
            for file in list_files:
                # filter for h5 files which are not hidden
                if '.h5' in file and '.' != file[0] and not file.split("_")[field_loc] in known_fields:
                    files.append(file)

            # walk through the filenames belonging to this experiment
            # and fetch all fields from filenames
            # !!! should be optimized, too much depending on correct naming
            for item in files:
                item = item[:-3]

                # don't add empty items if filename contains wrong fields
                new_field = item.split("_")[field_loc]
                if len(new_field) < 1:
                    continue
                try:
                    # grep all stimulus-names from filenames
                    stim_ = item.split("_")[stimulus_loc]
                    if stim_ == "512":
                        pass
                    else:
                        stimuli.append(stim_)
                        fields.append(new_field)
                except IndexError:
                    stimuli.append('')

                try:
                    # get from filename if field was recorded for its location
                    if "loc" in item.split("_")[stimulus_loc]:
                        locflag.append(1)
                    else:
                        locflag.append(0)
                except IndexError:
                    locflag.append(0)

                try:
                    # check filename if field is the optic disk
                    if "od" in item.split("_")[region_loc]:
                        odflag.append(1)
                    else:
                        odflag.append(0)
                except IndexError:
                    odflag.append(0)

                ### recording_depth wasn't working, so it is commented out
                # try:
                #    #print(headerpath + "Pre/" + item + '.h5')
                #    with h5py.File(headerpath + "Pre/" + item + '.h5', 'r') as file:
                #        depths[fields[-1]] = np.array(file['Depth'])
                # except KeyError:
                #    pass
            # recording_depths = []

            fields, idx = np.unique(fields, return_index=True)

            stimuli = [stimuli[i] for i in idx]
            locflag = [locflag[i] for i in idx]
            odflag = [odflag[i] for i in idx]

            # get data to populate subtable roi_mask from file
            fieldfiles, roi_masks, xabs, yabs, zabs = [], [], [], [], []
            for idx, item in enumerate(fields):
                # check for filename with entry belonging to correct stimulus and field
                filename = ""
                for f in files:
                    if stimuli[idx] in f and item in f:
                        filename = f
                        break

                # read all params from file
                fieldfiles.append(filename)
                with h5py.File(headerpath + "Pre/" + filename, 'r', driver="stdio") as f:
                    igor = list(f["wParamsNum"].attrs['IGORWaveDimensionLabels'][1:])
                    # print(f["wParamsNum"].attrs['IGORWaveDimensionLabels'][1:])
                    try:
                        absx = np.copy(f["wParamsNum"][igor.index('XCoord_um')])
                        absy = np.copy(f["wParamsNum"][igor.index('YCoord_um')])
                        absz = np.copy(f["wParamsNum"][igor.index('ZCoord_um')])
                        zstep = np.copy(f["wParamsNum"][igor.index('ZStep_um')])
                        zstack = np.copy(f["wParamsNum"][igor.index('User_ScanType')])
                    except ValueError:
                        absx = np.copy(f["wParamsNum"][igor.index(b'XCoord_um')])
                        absy = np.copy(f["wParamsNum"][igor.index(b'YCoord_um')])
                        absz = np.copy(f["wParamsNum"][igor.index(b'ZCoord_um')])
                        zstep = np.copy(f["wParamsNum"][igor.index(b'ZStep_um')])
                        zstack = np.copy(f["wParamsNum"][igor.index(b'User_ScanType')])

                    # print(zstack)
                    xabs.append(absx)
                    yabs.append(absy)
                    zabs.append(absz)

                    roi_masks.append(np.zeros((2, 2)))  # default, if missing
                    for h5_keys in f.keys():
                        if h5_keys.lower() == 'rois':  # different cases in different files...
                            roi_masks[-1] = np.copy(f[h5_keys])

                    ### recording_depth wasn't working, so it is commented out
                    # get ROI depths
                    # try:
                    #    recording_depths.append(depths[item])
                    # except KeyError:
                    #    recording_depths.append(np.array([]))

            # subkey for adding Fields to RoiMask
            subkey = key.copy()
            # subkey1 for adding Fields to ZStack
            subkey1 = key.copy()

            # iterate over fields, and add each for ZStack and RoiMask
            for i in range(len(fields)):
                idx = (Field() & key).fetch('field_id')
                if idx.size == 0:
                    field_id = 1
                else:
                    field_id = max(idx) + 1

                key["field_id"] = field_id
                key["field"] = fields[i]
                key["odflag"] = odflag[i]
                key["locflag"] = locflag[i]

                # extra info that shouldn't be in key
                # more than one insert for one experiment,
                # we need copy of key for multiple inserts
                key2 = key.copy()

                ### recording_depth wasn't working, so it is commented out
                # if recording_depths[i] != []:
                #    key2['recording_depth'] = recording_depths[i]

                # check "User_ScanType" of zstack
                if zstack == 11:
                    subkey1["zstack"] = 1
                else:
                    subkey1["zstack"] = 0

                subkey1["field_id"] = field_id
                subkey1["zstep"] = zstep

                subkey["absx"] = xabs[i]
                subkey["absy"] = yabs[i]
                subkey["absz"] = zabs[i]
                subkey["field_id"] = field_id
                subkey["fromfile"] = fieldfiles[i]
                subkey["roi_mask"] = roi_masks[i]

                ###
                # put location code here ???
                ###

                # print(key)
                # print(key2)
                # print(subkey)

                # add one Field with a copy of experiment parameters
                # and subkeys for RoiMask and ZStack Fields
                self.insert1(key2, allow_direct_insert=True)
                (Field().RoiMask() & key).insert1(subkey, allow_direct_insert=True)
                (Field().Zstack() & key).insert1(subkey1, allow_direct_insert=True)
        else:
            print("Error: Could not read path: " + headerpath + "Pre")


@schema
class Stimulus(dj.Manual):
    definition = """
    # Light stimuli

    stim_id             :tinyint            # Unique integer identifier
    ---
    framerate           :float              # framerate in hz
    stimulusname        :varchar(255)       # string identifier
    stimulus_trace      :longblob           # 2d or 3d array of the stimulus
    is_colour           :tinyint            # is stimulus coloured (eg, noise X cnoise )
    stim_path           :varchar(255)       # Path to hdf5 file containing numerical array and info about stim
    commit_id           :varchar(255)       # Commit id corresponding to stimulus entry in Github repo
    alias               :varchar(9999)      # Strings (_ seperator) used to identify this stimulus
    """

    class BlueGreenCS(dj.Part):
        definition = """
        # Blue and green center-surround stimulus
        ->Stimulus
        ---
        ntrials             :smallint unsigned  #number of times the stimulus is presented
        timeon_s            :float              #time that the stimulus is on for each condition
        timeoff_s           :float              #time that the stimulus is off for each condition
        dxstim_center_um    :smallint unsigned  #size of the center circle
        dxstim_surround_um  :smallint unsigned  #size of the surround (800 = full field)
        color_order         :blob               #list of strings with the order that the colors are presented
        location            :blob               #location of stimulus (center or surround). List of strings.
        """

    class MovingBar(dj.Part):
        definition = """
        # Moving bar stimulus, in 8 directions
        ->Stimulus
        ---
        ntrials             :tinyint unsigned                    # number of repetitions of the stimulus
        dirlist             :blob                                # direction list for moving bar type stimuli
        velocity            :float                               # velocity in um/second for moving stimuli
        t_move_dur_s        :float                               # amount of time bar is displayed
        bar_dx_um           :smallint unsigned                   # size of object in X
        bar_dy_um           :smallint unsigned                   # size of object in Y
        """

    class Chirp(dj.Part):
        definition = """
        # Standard global chirp stimulus
        ->Stimulus
        ---
        ntrials               :tinyint unsigned                  # number of repetitions of the stimulus
        chirpdur_s            :float      	                     # Rising chirp phase
        chirp_max_freq        :float                             # Peak frequency of chirp (Hz)
        contrast_freq         :float                             # Frequency at which contrast is modulated
        t_steady_off_s        :float                             # Light OFF at beginning
        t_steady_off2_s       :float                             # Light OFF at the end of stimulus
        t_steady_on_s         :float                             # Light 100 percent ON before and after chirp
        t_steady_mid_s        :float                             # Light at 50 percent for steps
        dx_stim_um            :smallint unsigned                 # Stimulus size
        """

    class Sine(dj.Part):
        definition = """
        # Sine modulation for center and surround with blue/green and other color at 60 percent contrast
        ->Stimulus
        ---
        ntrials               :tinyint unsigned                 # number of repetitions of the stimulus
        off_dur_s             :float                            # time that the stimulus is off for each condition
        on_dur_s              :float                            # time that the stimulus is on for each condition
        sine_dur_s            :float                            # time that the sine is on for each condition
        dx_stim_center_um     :smallint unsigned                # size of the center circle
        dx_stim_surround_um   :smallint unsigned                # size of the surround
        frequency             :float                            # frequency (1 Hz)
        color_order           :blob                             # list of strings with the order that the colors are presented
        location              :blob                             # list of strings with the location of stimulus (center or surround)
        """

    class BlueRing(dj.Part):
        definition = """
        # Flicker blue center/ring/surround
        ->Stimulus
        ---
        ntrials               :tinyint unsigned                 # number of repetitions of the stimulus
        dur_stim_s            :float                            # frequency (5 or 10 Hz)
        dx_stim_center_um     :smallint unsigned                # size of the center circle
        dx_stim_ring_um       :smallint unsigned                # size of the ring
        dx_stim_surround_um   :smallint unsigned                # size of the surround
        fname_noise_cblue     :varchar(255)                     # file name for center noise
        fname_noise_rblue     :varchar(255)                     # file name for ring noise
        fname_noise_sblue     :varchar(255)                     # file name for surround noise
        """

    class GreenRing(dj.Part):
        definition = """
        # Flicker green center/ring/surround
        ->Stimulus
        ---
        ntrials               :tinyint unsigned                 # number of repetitions of the stimulus
        dur_stim_s            :float                            # frequency (5 or 10 Hz)
        dx_stim_center_um     :smallint unsigned                # size of the center circle
        dx_stim_ring_um       :smallint unsigned                # size of the ring
        dx_stim_surround_um   :smallint unsigned                # size of the surround
        fname_noise_cgreen    :varchar(255)                     # file name for center noise
        fname_noise_rgreen    :varchar(255)                     # file name for ring noise
        fname_noise_sgreen    :varchar(255)                     # file name for surround noise
        """

    class BlueGreenRing(dj.Part):
        definition = """
        # Flicker with simultaneous presentation of blue and green center/ring/surround and test sequences
        ->Stimulus
        ---
        ntrials               :tinyint unsigned                 # number of repetitions of the stimulus
        dur_stim_s            :float                            # frequency (5 or 10 Hz)
        dx_stim_center_um     :smallint unsigned                # size of the center circle
        dx_stim_ring_um       :smallint unsigned                # size of the ring
        dx_stim_surround_um   :smallint unsigned                # size of the surround
        fname_noise_cblue     :varchar(255)                     # file name for center noise
        fname_noise_rblue     :varchar(255)                     # file name for ring noise
        fname_noise_sblue     :varchar(255)                     # file name for surround noise
        fname_noise_cgreen    :varchar(255)                     # file name for center noise
        fname_noise_rgreen    :varchar(255)                     # file name for ring noise
        fname_noise_sgreen    :varchar(255)                     # file name for surround noise
        test_sequence_lines   :tinyblob                         # which lines of stimulus were included in test sequence, list of [start, stop]
        main_start_lines      :blob                             # line of noise array where main sequences start
        main_end_lines        :blob                             # line of noise array where main sequences end
        fintenw_b             :smallint unsigned                # intensity of the blue channel
        fintenw_g             :smallint unsigned                # intensity of the green channel
        """

    class BlueGreenFlicker(dj.Part):
        definition = """
        # Flicker stimulus with separate epochs of blue and green flicker, test sequences, and breaks between stimuli
        ->Stimulus
        ---
        ntrials                 :tinyint unsigned       # number of repetitions of the stimulus
        dur_stim_s              :float                  # frame period of stimulation
        trigger_freq            :float                  # period between triggers in seconds
        dx_stim_center_um       :smallint unsigned      # size of the center circle
        dx_stim_ring_um         :smallint unsigned      # size of the ring
        dx_stim_surround_um     :smallint unsigned      # size of the surround
        fname_noise_cblue       :varchar(255)           # file name for center noise
        fname_noise_rblue       :varchar(255)           # file name for ring noise
        fname_noise_sblue       :varchar(255)           # file name for surround noise
        fname_noise_cgreen      :varchar(255)           # file name for center noise
        fname_noise_rgreen      :varchar(255)           # file name for ring noise
        fname_noise_sgreen      :varchar(255)           # file name for surround noise
        fname_noise_ctest       :varchar(255)           # file name for center test sequence
        fname_noise_stest       :varchar(255)           # file name for the surround test sequence
        fname_noise_rtest       :varchar(255)           # file name for the ring test sequence
        dur_break               :float                  # duration of break between epochs in seconds
        fintenw_b               :smallint unsigned      # intensity of the blue channel
        fintenw_g               :smallint unsigned      # intensity of the green channel
        epoch_start_lines       :tinyblob               # list of lines where epochs start
        epoch_end_lines         :tinyblob               # list of lines where epochs end
        n_fr_test               :smallint unsigned      # number of frames of test sequence
        test_seq_triggers       :tinyblob               # list of trigger frames in the test sequence
        order_string            :varchar(255)           # sequence of stimulus
        """

    class Blue_CSFlicker(dj.Part):
        definition = """
        # Flicker blue center/surround used in Szatko et al. 2020 for bipolar cells
        ->Stimulus
        ---
        ntrials               :tinyint unsigned                 # number of repetitions of the stimulus
        dur_stim_s            :float                            # frequency (5 or 10 Hz)
        dx_stim_center_um     :smallint unsigned                # size of the center circle
        dx_stim_surround_um   :smallint unsigned                # size of the surround
        fname_noise_cblue     :varchar(255)                     # file name for center noise
        fname_noise_sblue     :varchar(255)                     # file name for surround noise
        """

    class Green_CSFlicker(dj.Part):
        definition = """
        # Flicker green center/surround used in Szatko et al. 2020 for bipolar cells
        ->Stimulus
        ---
        ntrials               :tinyint unsigned                 # number of repetitions of the stimulus
        dur_stim_s            :float                            # frequency (5 or 10 Hz)
        dx_stim_center_um     :smallint unsigned                # size of the center circle
        dx_stim_surround_um   :smallint unsigned                # size of the surround
        fname_noise_cgreen    :varchar(255)                     # file name for center noise
        fname_noise_sgreen    :varchar(255)                     # file name for surround noise
        """


@schema
class Presentation(dj.Computed):
    definition = """
    # information about each stimulus presentation
    -> Field
    -> Stimulus
    # -> Pharmacology

    presentation_id     :int            # automatic surrogate id; auto_increment?
    ---
    h5_header           :varchar(255)   # path to h5 file
    triggertimes        :longblob       # triggertimes in each presentation
    triggervalues       :longblob       # values of the recorded triggers
    scan_line_duration  :float          # duration of one line scan
    scan_num_lines      :float          # number of scan lines (in XZ scan)
    scan_frequency      :float          # effective sampling frequency for each pixel in the scan field
    """

    class ScanInfo(dj.Part):
        definition = """
        #meta data recorded in scamM header file
        -> Presentation
        ---

        hdrleninvaluepairs         :float                      #
        hdrleninbytes              :float                      #
        minvolts_ao                :float                      #
        maxvolts_ao                :float                      #
        stimchanmask               :float                      #
        maxstimbufmaplen           :float                      #
        numberofstimbufs           :float                      #
        targetedpixdur_us          :float                      #
        minvolts_ai                :float                      #
        maxvolts_ai                :float                      #
        inputchanmask              :float                      #
        numberofinputchans         :float                      #
        pixsizeinbytes             :float                      #
        numberofpixbufsset         :float                      #
        pixeloffs                  :float                      #
        pixbufcounter              :float                      #
        user_scanmode              :float                      #
        user_dxpix                 :float                      #
        user_dypix                 :float                      #
        user_npixretrace           :float                      #
        user_nxpixlineoffs         :float                      #
        user_nypixlineoffs         :float                      # update 20171113
        user_divframebufreq        :float                      #
        user_scantype              :float                      #
        user_scanpathfunc          :varchar(255)               #
        user_nsubpixoversamp       :float                      #
        user_nfrperstep            :float                      #
        user_xoffset_v             :float                      #
        user_yoffset_v             :float                      #
        user_offsetz_v             :float                      #
        user_zoomz                 :float                      # update 20171113
        user_noyscan               :float                      # update 20171113
        realpixdur                 :float                      #
        oversampfactor             :float                      #
        xcoord_um                  :float                      #
        ycoord_um                  :float                      #
        zcoord_um                  :float                      #
        zstep_um                   :float                      #
        zoom                       :float                      #
        angle_deg                  :float                      #
        datestamp_d_m_y            :varchar(255)               #
        timestamp_h_m_s_ms         :varchar(255)               #
        inchan_pixbuflenlist       :varchar(255)               #
        username                   :varchar(255)               #
        guid                       :varchar(255)               #
        origpixdatafilename        :varchar(255)               #
        stimbuflenlist             :varchar(255)               #
        callingprocessver          :varchar(255)               #
        callingprocesspath         :varchar(255)               #
        targetedstimdurlist        :varchar(255)               #
        computername               :varchar(255)               #
        scanm_pver_targetos        :varchar(255)               #
        user_zlensscaler           :float                      #
        user_stimbufperfr          :float                      #
        user_aspectratiofr         :float                      #
        user_zforfastscan          :float                      #
        user_zlensshifty           :float                      #
        user_nzpixlineoff          :float                      #
        user_dzpix                 :float                      #
        user_setupid               :float                      #
        user_nzpixretrace          :float                      #
        user_laserwavelen_nm       :float                      #
        user_scanpathfunc          :varchar(255)               #
        user_dzfrdecoded           :float                      #
        user_dxfrdecoded           :float                      # update 20171113
        user_dyfrdecoded           :float                      # update 20171113
        user_zeroz_v               :float                      #
        igorguiver                 :varchar(255)               #
        user_comment               :varchar(255)               #
        user_objective             :varchar(255)               #
        realstimdurlist=""         :varchar(255)               # update 20180529
        user_ichfastscan           :float                      # update 20171113
        user_trajdefvrange_v       :float                      # update 20171113
        user_ntrajparams           :float                      # update 20171113
        user_offset_v              :float                      # update 20180529
        user_etl_polarity_v        :float                      # update 20180529
        user_etl_min_v             :float                      # update 20180529
        user_etl_max_v             :float                      # update 20180529
        user_etl_neutral_v         :float                      # update 20180529
        user_nimgperfr             :float                      # update 20180529
        user_warpparamslist        :varchar(255)               # update 20200407
        user_nwarpparams           :float                      # update 20200407
        """

    def _make_tuples(self, key):
        # !!! Table depends on naming convention too strongly, e.g. RR1 needed not RR. Needs to be changed.

        # Get all params needed for this function
        date = key["date"]
        date = date.strftime('%Y-%m-%d')
        # print(date)
        stim = (Stimulus() & key).fetch1("stimulusname")
        stim_alias = (Stimulus() & key).fetch1("alias").split('_')

        exp_num = key["exp_num"]
        field = key["field_id"]
        field_str = (Field() & key).fetch1("field")

        stim_loc = (UserInfo() & key).fetch1("stimulus_loc")
        field_loc = (UserInfo() & key).fetch1("field_loc")
        headerpath = (Experiment() * Field() & key).fetch1("headerpath")

        # copy for inserting subkey, is this needed ?
        primarykey = key.copy()

        # check if files exists and the filter for valid files
        if os.path.exists(headerpath + "Pre"):
            list_files = os.listdir(headerpath + "Pre")
            files = []
            for file in list_files:
                # filter for h5 files which are not hidden and with correct field string in filename
                if '.h5' in file and '.' != file[0] and field_str in file.split("_")[field_loc]:
                    files.append(file)

            # iterate over valid files and add presentations from one experiment
            for item in files:

                item = item[0:item.find(".h5")]
                split_string = item.split("_")

                # search for stim or nostim
                if stim_loc < len(split_string):
                    stim = split_string[stim_loc]
                else:
                    stim = 'nostim'

                # print(item)

                # already correct files filtered above, only check for correct stim
                if (stim.lower() in stim_alias):  # careful this can cause errors. Might want to change to "==" AV

                    # increment index of presentation or start with 1
                    idx = (Presentation() & 'exp_num={0} AND date="{1}" '.format(exp_num, date)).fetch(
                        'presentation_id')
                    if idx.size == 0:
                        idx = 1
                    else:
                        idx = max(idx) + 1

                    key['presentation_id'] = idx
                    primarykey['presentation_id'] = idx
                    filepath = headerpath + "Pre/" + item + ".h5"

                    # create subkey from copy of key, see above, is this needed ?
                    subkey = primarykey.copy()
                    key["h5_header"] = filepath

                    # print(filepath)

                    # load files with driver:"stdio", faster then default driver
                    # [()] reads the whole file to memory not only chunks (bit faster)
                    with h5py.File(filepath, 'r', driver="stdio") as f:
                        if "Triggertimes" in f.keys():  # case sensitive?
                            key["triggertimes"] = f["Triggertimes"][()]
                            key["triggervalues"] = f["Triggervalues"][()]
                        elif "triggertimes" in f.keys():  # case sensitive?
                            key["triggertimes"] = f["triggertimes"][()]
                            key["triggervalues"] = f["Triggervalues"][()]
                        elif "TriggerTimes" in f.keys():
                            key["triggertimes"] = f["TriggerTimes"][()]
                            key["triggervalues"] = f["Triggervalues"][()]
                        else:
                            key["triggertimes"] = np.zeros(1)
                            key["triggervalues"] = np.zeros(1)
                        # get scanning frequency
                        try:
                            roi_mask = (Field.RoiMask() & primarykey).fetch('roi_mask')[0]
                            os_params = list(
                                zip(f['OS_Parameters'].attrs['IGORWaveDimensionLabels'][1:], f['OS_Parameters']))
                            try:
                                key["scan_line_duration"] = \
                                    [v for k, v in os_params if k[0].decode() == 'LineDuration'][0]
                            except AttributeError:  # if string not encoded
                                key["scan_line_duration"] = [v for k, v in os_params if k[0] == 'LineDuration'][0]
                            key["scan_num_lines"] = roi_mask.shape[-1]
                            key["scan_frequency"] = np.round(
                                1 / (key["scan_line_duration"] * key["scan_num_lines"]), 2)
                        except KeyError:
                            key["scan_line_duration"] = -1
                            key["scan_num_lines"] = -1
                            key["scan_frequency"] = -1
                    # add new presentation and give output to see process
                    self.insert1(key)
                    print("added new Presentation for: " + filepath)

                    ### extract params for subkey and add it with copy of key "primarykey"
                    subkey = self.extract_from_h5(subkey, ["wParamsStr", "wParamsNum"], filepath)
                    # Take care of exceptions generated when scanm is updated and new parameters are added
                    # Maybe set default values instead.
                    # LR: It is surprising that there needs to be such a hardcoded failsafe here?
                    subkey = self.fill_missing(subkey,
                                               ['user_zlensshifty', 'user_dzfrdecoded', 'user_zeroz_v',
                                                'user_zoomz',
                                                'user_nypixlineoffs', 'user_offsetz_v', 'user_dxfrdecoded',
                                                'user_dyfrdecoded',
                                                'user_ichfastscan', 'user_noyscan', 'user_ntrajparams',
                                                'user_trajdefvrange_v',
                                                'user_zlensshifty', 'user_stimbufperfr', 'user_aspectratiofr',
                                                'zstep_um',
                                                'user_zlensscaler', 'user_zforfastscan', 'user_nzpixlineoff',
                                                'user_dzpix',
                                                'user_nzpixretrace', 'user_etl_polarity_v', 'user_offset_v',
                                                'user_zeroz_v',
                                                'user_etl_polarity_v', 'user_etl_min_v', 'user_etl_max_v',
                                                'user_etl_neutral_v',
                                                'user_setupid', 'user_laserwavelen_nm', 'user_comment',
                                                'user_objective', 'user_nimgperfr', 'user_warpparamslist',
                                                'user_nwarpparams'
                                                ])

                    (Presentation().ScanInfo() & primarykey).insert1(subkey)
        else:
            print("Could not read path: " + headerpath + "Pre")

    def fill_missing(self, key1, stringlist):
        # if items are empty set them to "0"
        for item in stringlist:
            if not item in key1:
                key1[item] = 0
            elif item == "stimbufperfr" or item == "aspectratiofr":
                print("exist")
        return key1

    def extract_from_h5(self, key1, datasetstringlist, filepath):
        # load files with driver:"stdio", faster then default driver
        with h5py.File(filepath, 'r', driver="stdio") as f:
            # fetch items from file dataset, decode them if "type == bytes"
            # and then set the values of the items to key1
            for dataset in datasetstringlist:
                item_list_fromfile = (list(zip(f[dataset].attrs['IGORWaveDimensionLabels'][1:], f[dataset])))
                for item in item_list_fromfile:
                    item_key = item[0][0]
                    if type(item_key) == bytes:
                        item_key = item_key.decode()
                    if item_key is not "":
                        value = item[1]
                        if type(value) == bytes:
                            value = value.decode()
                        key1[item_key.lower()] = value
        # print(key1)
        return key1


@schema
class Pharmacology(dj.Computed):
    definition = """
    # information about pharmacological treatments
    -> Presentation
    -> PharmInfo
    ---
    treatment           :varchar(255)       # string of the treatment name from hdf5 file
    control_flag        :int                # 1 if control 0 if in drug
    concentration       :varchar(255)       # drug concentration(s), "0" for controls
    multiple_drug_flag  :int                # 1 if multiple drugs, else 0
    """

    @property
    def key_source(self):
        rel = super().key_source
        return rel * (
                PharmInfo() & 'pharmaflag = 1').proj()  # rel * (Stimulus & 'stimulusname in ("bluering", "greenring", "bluegreenring")').proj()

    def _make_tuples(self, key):
        # get the information
        pharm_loc = (UserInfo() * Presentation() & key).fetch1("pharm_loc")
        h5_header = (Presentation & key).fetch1("h5_header")
        filename = h5_header.split("/")[-1]
        txt = filename.split("_")[pharm_loc].split(".")[0]
        pharm_info = (PharmInfo() & key).fetch1()

        # binary of whether it's control or treated
        if txt.lower() == "ctrl":  # warning, this has to be written exactly this way in the hdf5 file name
            control = 1
            treatment = "control"
            concentration = "0"
            multiple_drug = 0
        elif ';' in pharm_info['drug']:
            control = 0
            treatment = pharm_info['drug']
            concentration = pharm_info['pharmconc']
            multiple_drug = 1
        else:
            control = 0
            treatment = pharm_info['drug']
            concentration = pharm_info['pharmconc']
            multiple_drug = 0

        # add to the table
        Pharmacology().insert1(
            dict(key,
                 treatment=treatment,
                 control_flag=control,
                 concentration=concentration,
                 multiple_drug_flag=multiple_drug
                 ))


@schema
class Traces(dj.Computed):
    definition = """
    # Raw Traces for each roi under a specific presentation

    -> Presentation

    ---
    traces                   :longblob              # array of raw traces (frame x roi)
    traces_times             :longblob              # numerical array of trace times (frame x roi)
    traces_flag              :tinyint               # flag if values in traces are correct(1) or not(0)
    """

    @property
    def key_source(self):
        return Presentation()  # & 'date="2016-02-16" AND exp_num=1 AND field_id=1'

    def _make_tuples(self, key):

        # get all params we need for creating traces
        filepath = (Presentation() & key).fetch1("h5_header")

        # load files with driver:"stdio", faster then default loading
        # [()] reads the whole file to memory not only chunks (bit faster)
        with h5py.File(filepath, "r", driver="stdio") as f:
            # read all traces and their times from file
            if "Traces0_raw" in f.keys() and "Tracetimes0" in f.keys():
                traces = f["Traces0_raw"][()]
                traces_times = f["Tracetimes0"][()]
                traces_dim = len(traces.shape)
            else:
                # set dim = -1 if traces couldn't be read
                traces_dim = -1

            # assign data to key
            if traces_dim == 2:
                key["traces"] = traces
                key["traces_times"] = traces_times
                key["traces_flag"] = 1
            else:
                # ... if not initialize with default values
                key["traces"] = []
                key["traces_times"] = []
                key["traces_flag"] = 0

            self.insert1(key)
            print("populated for experimenter {}, date {}, exp {}, field {}"
                  .format(key["experimenter"], key["date"], key["exp_num"],
                          key["field_id"]))


@schema
class TraceTriggerTimes(dj.Computed):
    definition = """
    # Corrected for Igor OS script error by removing the erroneously added stimulator delay and correcting the trigger times for stimulator delay
    -> Traces
    ---
    trace_times                :longblob               # trace times in time x roi
    trigger_times_corrected    :mediumblob             # trigger times adjusted for stimulator delay
    """

    def _make_tuples(self, key):
        #get the original (incorrect) trace times imported from Igor
        uncorrected_trace_times = (Traces & key).fetch1('traces_times')

        #get the stimulator delay that was used in Igor
        path = (Experiment() & key).fetch1('headerpath')
        path = path + 'Pre/'
        file = (Field.RoiMask() & key).fetch1('fromfile')
        datafile = h5py.File(path + file, 'r')
        os_parameters = np.copy(datafile['OS_Parameters'])
        stimulator_delay = os_parameters[8] / 1000  # converting from ms to seconds
        datafile.close()

        #get the trigger times
        triggertimes = (Presentation() & key).fetch1('triggertimes')

        #update the trace times
        trace_times = uncorrected_trace_times - stimulator_delay #note: now the trace times are in "real time"

        #update the triggertimes
        trigger_times_corrected = triggertimes + stimulator_delay

        self.insert1(dict(key,
                        trace_times=trace_times,
                        trigger_times_corrected=trigger_times_corrected,
        ))


@schema
class PreprocessParams(dj.Lookup):
    definition = """
    preprocess_param_set_id     :tinyint                     # Unique parameter set ID
    ---
    high_pass_freq              :float                   # Cutoff frequency of highpass filter
    order                       :tinyint                     # Filter order
    standardize                 :enum("True", "False")   # Whether to z-score (subtract mean, divide by standard deviation)
    """


@schema
class PreprocessTraces(dj.Computed):
    definition = """
    # Preprocessed Traces
    -> PreprocessParams
    -> Traces
    -> Presentation
    ---
    preprocess_traces           :longblob                # Preprocessed traces (frame x roi)
    stim_beginning_frame        :int                     # Frame when first trigger occurs
    preproc_flag                :tinyint                 # flag about whether preprocessing was successful (1 = yes, 0 = no)
    """

    # @property
    # def key_source(self):
    #     rel = super().key_source
    #     return rel * Presentation() - ['scan_frequency > 300']

    def _make_tuples(self, key):
        high_pass_freq = (PreprocessParams() & key).fetch1('high_pass_freq')
        order = (PreprocessParams() & key).fetch1('order')
        standardize = (PreprocessParams() & key).fetch1('standardize')
        sampling_freq = (Presentation() & key).fetch1('scan_frequency')
        raw_traces = (Traces() & key).fetch1('traces')
        nyq = 0.5 * sampling_freq
        normal_cutoff = high_pass_freq / nyq
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)  # High pass filter

        n_rois = raw_traces.shape[1]
        n_frames = raw_traces.shape[0]

        traces_preproc = np.copy(raw_traces)

        try:
            for roi in range(n_rois):
                traces_preproc[:, roi] = signal.filtfilt(b, a, raw_traces[:, roi])  # Filtered traces

            stim_beginning = (Presentation() & key).fetch1('triggertimes')[0]  # First trigger, time based
            nb_lines = (Presentation() & key).fetch1('scan_num_lines')  # Number of lines
            line_duration = (Presentation() & key).fetch1('scan_line_duration')  # Duration of scanning one line
            frame_duration = nb_lines * line_duration  # Duration of scanning one frame
            stim_beginning_frame = math.ceil(stim_beginning / frame_duration)  # Calculate first trigger, frame based

            if standardize == "True":
                norm_period = traces_preproc[
                              stim_beginning_frame:,
                              :]  # Getting all data after first trigger occured, time period used for normalization
                norm_period_mean = np.mean(norm_period, axis=0)
                norm_period_sd = np.std(norm_period, axis=0)

                for roi in range(n_rois):
                    traces_preproc[:, roi] = (traces_preproc[:, roi] - norm_period_mean[roi]) / norm_period_sd[
                        roi]  # Fully preprocessed traces

            preproc_flag = 1
            self.insert1(dict(key, preprocess_traces=traces_preproc, stim_beginning_frame=stim_beginning_frame,
                              preproc_flag=preproc_flag))
            print("populated for experimenter {}, date {}, exp {}, field {}"
                  .format(key["experimenter"], key["date"], key["exp_num"],
                          key["field_id"]))
        except Exception as err:
            print("an error occurred during preprocessing:", err)
            print(key, "not preprocessed")
            traces_preproc = []
            stim_beginning_frame = -1
            preproc_flag = 0
            self.insert1(dict(key, preprocess_traces=traces_preproc, stim_beginning_frame=stim_beginning_frame,
                              preproc_flag=preproc_flag))


@schema
class PreprocessSnippets(dj.Computed):
    definition = """
    # Snippet dataframe created from slicing filtered traces using the triggertimes.

    -> Presentation
    -> PreprocessTraces
    -> Traces
    -> TraceTriggerTimes
    ---
    preprocess_snippets             :longblob               # array of snippets
    preprocess_snippets_times       :longblob               # array of snippet times
    trial_nums                      :longblob               # array to align to get trial values
    preprocess_snippets_x_times     :longblob               # time relative to start of trigger for trials
    roi_ids                         :longblob               # array to align to get roi id numbers

    """

    class SnippetConditions(dj.Part):
        definition = """
        # for including condition information with the snippets

        -> PreprocessSnippets
        -> Stimulus

        ---
        stim_condition_strs         :varchar(255)              # string of condition labels in correct order
        stim_conditions             :longblob                  # array to align to get conditions for each trial

    """

    @property  # this is how you exclude!!
    def key_source(self):
        rel = super().key_source

        return rel * (
                Stimulus() - 'stimulusname in ("no_stim", "bluering", "greenring","bluegreenring", "bluegreenflicker")').proj() * (
                       PreprocessTraces() - 'preproc_flag in (0)').proj()

    def _make_tuples(self, key):
        # print("Populating for ", key) #we presumably don't want it to do this for every single ROI...

        # first pick triggermode. Ours should always be 1 (except for chirp)
        stim_id = (Presentation() & key).fetch1('stim_id')
        if stim_id in [5, 6]:  # chirp stimuli (including chirpall)
            nth_snippet = 2
        # elif stim_id == 8:  # ring 1D stimulus, each direction repeats 4x before changing
        #     nth_snippet = 4
        else:
            #     # print('Add a new case for stimulus with stim_id ', str(stim_id),
            #     #       '; for now assuming every trigger should be used')
            nth_snippet = 1

        # fetch data
        triggertimes = (TraceTriggerTimes() & key).fetch1('trigger_times_corrected')
        traces_times_all = (TraceTriggerTimes() & key).fetch1('trace_times')
        preprocess_traces_all = (PreprocessTraces() & key).fetch1('preprocess_traces')

        # initialize dataframe
        columns = ['snippets', 'snippets_times', 'x_times', 'trial_nums', 'roi_nums']
        data_df = pd.DataFrame(columns=columns)

        n_rois = preprocess_traces_all.shape[1]

        # deal with trigger mode and find trial durations
        valid_triggers = np.copy(triggertimes[::nth_snippet])
        trial_durations = np.diff(valid_triggers)
        max_trial_dur = np.max(trial_durations)
        trial_durations = np.concatenate((trial_durations, max_trial_dur),
                                         axis=None)  # last trial uses the maximum trial duration
        trial_ends = valid_triggers + trial_durations  # trials are as long as each trial's true duration
        stimulus_start = triggertimes[0]
        stimulus_stop = trial_ends[-1]

        if stim_id in [0, 1]: #we are doing this to have a baseline!!!
            time_shift = 0.2
            stimulus_start = stimulus_start - time_shift  # adding buffer time before trigger for bluegreenCS
            stimulus_stop = stimulus_stop - time_shift
            valid_triggers = valid_triggers - time_shift
            trial_ends = trial_ends - time_shift

        for roi in range(n_rois):
            # get trace and triggertimes
            traces_times = traces_times_all[:, roi]
            preprocess_traces = preprocess_traces_all[:, roi]

            # truncate traces and trace times to only include the stimulated period

            time_mask = (traces_times > stimulus_start) & (traces_times < stimulus_stop)
            snippets = preprocess_traces[time_mask]
            snippets_times = traces_times[time_mask]

            # initialize the delta x, trial number arrays, and roi_nums array
            x_times = snippets_times
            trial_nums = (-10000) * np.ones(snippets_times.shape)
            roi_num = np.ones(snippets_times.shape) * (roi + 1)

            # update trial numbers and delta x times
            for trial_num, time in enumerate(valid_triggers):
                trial_mask = (snippets_times > time) & (snippets_times <= trial_ends[trial_num])
                # assign trials based on start of each trigger
                trial_nums = np.where(trial_mask, trial_num, trial_nums)
                x_times = np.where(trial_mask, (x_times - time), x_times)

            # add data to dataframe
            data_stack = np.stack([snippets, snippets_times, x_times, trial_nums, roi_num], axis=1)
            data_df = pd.concat([data_df, pd.DataFrame(data_stack, columns=columns)])

        if stim_id in [0, 1]:  # setting triggers to zero still
            data_df['x_times'] = data_df['x_times'] - time_shift

        # indexing
        data_df = data_df.astype({'roi_nums': 'int32', 'trial_nums': 'int32'})
        data_df = data_df.set_index(['roi_nums', 'trial_nums'])
        # drop data between trials
        if (-10000) in data_df.index:
            data_df = data_df.drop(
                (-10000))  # drop df rows that are outside of the stimulus duration (for uneven trial length)

        # reassign arrays and insert into table
        data_df = data_df.reset_index()
        snippets = data_df['snippets'].to_numpy()
        snippets_times = data_df['snippets_times'].to_numpy()
        x_times = data_df['x_times'].to_numpy()
        trial_nums = data_df['trial_nums'].to_numpy()
        roi_nums = data_df['roi_nums'].to_numpy()

        self.insert1(dict(key,
                          preprocess_snippets=snippets,
                          preprocess_snippets_times=snippets_times,
                          trial_nums=trial_nums,
                          preprocess_snippets_x_times=x_times,
                          roi_ids=roi_nums)
                     )

        # insert conditions into the part table if they are supported
        stimname = (Stimulus() & key).fetch1('stimulusname')
        supported_stim_list = ['bluegreencs', 'movingbar', 'sine']
        primary_key = deepcopy(key)

        if stimname in supported_stim_list:
            data_df = data_df.set_index(['roi_nums', 'trial_nums'])

            if stimname == "bluegreencs":
                # fetch parameters
                colors, locations, n_trials_stim = (Stimulus.BlueGreenCS() & key).fetch1(
                    'color_order', 'location', 'ntrials'
                )
                # n_trials = np.amax(trial_nums)
                # if n_trials_stim != n_trials:
                #     #figure out what to do here later. Does it actually matter?

                # first insert directions
                condvals1 = list(colors)
                condname1 = 'colors'
                level1 = 1
                data_df = self.insertcondition(data_df, condvals1, condname1, level1)

                # then insert lengths
                condvals2 = list(locations)
                condname2 = 'locations'
                level2 = 2
                data_df = self.insertcondition(data_df, condvals2, condname2, level2)

                data_df = data_df.reset_index()
                colors = data_df['colors']  # .to_numpy().astype(int)
                locations = data_df['locations']  # .to_numpy().astype(int)
                conditions = np.column_stack((colors, locations))
                stim_cond_strs = 'colors locations'

            if stimname == "sine":
                # fetch parameters
                colors, locations, n_trials_stim = (Stimulus.Sine() & key).fetch1(
                    'color_order', 'location', 'ntrials'
                )
                # n_trials = np.amax(trial_nums)
                # if n_trials_stim != n_trials:
                #     #figure out what to do here later. Does it actually matter?

                # first insert directions
                condvals1 = list(colors)
                condname1 = 'colors'
                level1 = 1
                data_df = self.insertcondition(data_df, condvals1, condname1, level1)

                # then insert lengths
                condvals2 = list(locations)
                condname2 = 'locations'
                level2 = 2
                data_df = self.insertcondition(data_df, condvals2, condname2, level2)

                data_df = data_df.reset_index()
                colors = data_df['colors']  # .to_numpy().astype(int)
                locations = data_df['locations']  # .to_numpy().astype(int)
                conditions = np.column_stack((colors, locations))
                stim_cond_strs = 'colors locations'

            if stimname == "movingbar":
                # fetch parameters
                dirlist, n_trials_stim = (Stimulus.MovingBar() & key).fetch1(
                    'dirlist', 'ntrials'
                )
                # n_trials = np.amax(trial_nums)
                # if n_trials_stim != n_trials:
                #     #figure out what to do here later. Does it actually matter?
                # first insert directions
                condvals1 = list(dirlist)
                condname1 = 'directions'
                level1 = 1
                data_df = self.insertcondition(data_df, condvals1, condname1, level1)

                data_df = data_df.reset_index()
                conditions = data_df['directions'].to_numpy().astype(int)
                stim_cond_strs = 'directions'

            PreprocessSnippets.SnippetConditions().insert1(
                dict(primary_key,
                     stim_condition_strs=stim_cond_strs,
                     stim_conditions=conditions
                     ))
        print("populated for experimenter {}, date {}, exp {}, field {}"
              .format(key["experimenter"], key["date"], key["exp_num"],
                      key["field_id"]))
        # if key['roi_id'] == 1:  # print the status just for the first ROI from each field
        #     print('Populated for experimenter {}, date {}, exp {}, field {}, '
        #           'roi {}'.format(key["experimenter"], key["date"], key["exp_num"], key["field_id"], key["roi_id"]))

    def insertcondition(self, datadf, conditionvalues, conditionname, level):
        # inserts a new index into your multiindexed data frame datadf at a specific level
        # requires an array of trial numbers sorted into rows containing trials that share a condition
        # and an ordered list of the condition values (ie degrees) in the order they were presented.
        # For inserting multiple conditions, run twice and make sure that levels and names are different

        # make array to map trials onto conditions
        alltrials = datadf.index.get_level_values('trial_nums').unique()
        nconditions = len(conditionvalues)
        ntrials_cond = (np.amax(alltrials) + 1) // nconditions
        trialconditions = np.transpose(np.resize(alltrials, (ntrials_cond, nconditions)))

        old_idx = datadf.index.to_frame()  # Convert index to dataframe
        conditionvals = old_idx["trial_nums"]  # have to deal with the problem of extra trials and/or repeats

        # altering trialconditions and conditionvals to prevent replacement of a condition with another
        conditionvals = conditionvals * 3000 - 1
        trialconditions = trialconditions * 3000 - 1

        xval = 0
        for row in trialconditions:
            conditionvals = conditionvals.replace(row, conditionvalues[xval])
            xval += 1
        conditionvals = np.asarray(conditionvals)
        old_idx.insert(level, conditionname, conditionvals)  # Insert new level at specified location
        datadf.index = pd.MultiIndex.from_frame(old_idx)  # Convert back to MultiIndex

        return datadf


@schema
class ResamplingFrequency(dj.Lookup):
    definition = """
    # Frequency for resampling and averaging
    resample_f_id             :int            # Unique parameter set ID
    ---
    resample_f                :int            # frequency for resampling in Hz
    """


@schema
class QualityParams(dj.Lookup):
    definition = """
    # parameters for determining the identity of high quality pixels (part class SummarySnippets.HighQualityPixels)

    quality_id          :int                # arbitrary integer identifier

    ---

    quality_method      :varchar(255)       # "percentile" or "hard_threshold"
    threshold           :float              # threshold percentile or quality threshold

    """


@schema
class SummarySnippets(dj.Computed):
    definition = """
    # Resampling, averaging, and maximums for snippets
    -> PreprocessSnippets
    -> Stimulus
    -> ResamplingFrequency
    ---
    resampled_time_bins       :longblob         # resampled time bins in same format as the rest of the snippet dataframe
    condition_average_frame   :longblob         # matrix of averages with conditions and times
    condition_average_strs    :varchar(255)     # list of string labels for average_frame columns
    condition_max_frame       :longblob         # matrix of maxima by trial
    condition_max_strs        :varchar(255)     # list of string labels for max_frame columns
    quality_indexes_frame     :longblob         # quality index (by condition if there are conditions)
    quality_indexes_strs      :varchar(255)     # list of string labels for quality_indexes_frame
    empty_quality_flag        :tinyint          # 1 if there are no high quality pixels, 0 otherwise
    """

    class HighQualityPixels(dj.Part):
        definition = """
        -> SummarySnippets
        -> QualityParams
        ---
        high_quality_pixels     :longblob           # list of high quality pixels (roi ids)
        """

    @property
    def key_source(self):
        # rel = super().key_source
        # return rel * Stimulus() - ['stimulusname = "no_stim"',
        #                            'stimulusname = "noise"'
        #                            ]
        rel = super().key_source
        return rel - (Stimulus & 'stimulusname in ("no_stim", "noise")').proj()

    def _make_tuples(self, key):

        # get data
        resamp_f = (ResamplingFrequency() & key).fetch1('resample_f')
        resamp_p = 1 / resamp_f
        data_df = pd.DataFrame((PreprocessSnippets() & key).fetch1())

        # resample data by binning the time traces
        data_df = data_df.drop(
            ['experimenter', 'date', 'exp_num', 'field_id', 'stim_id', 'presentation_id', 'preprocess_param_set_id'],
            axis=1)
        data_df['time_bin'] = data_df['preprocess_snippets_x_times'].apply(
            lambda x_t: resamp_p * (round(x_t / resamp_p)))

        data_df = data_df.astype({'roi_ids': 'int32', 'trial_nums': 'int32'})

        # get averages, maxima and quality values
        stimname = (Stimulus() & key).fetch1('stimulusname')
        supported_stim_list = ['bluegreencs', 'movingbar']  # MK

        if stimname in supported_stim_list:

            conditions, cond_strs = (PreprocessSnippets.SnippetConditions() & key).fetch1(
                'stim_conditions', 'stim_condition_strs')
            cond_list = cond_strs.split()
            if len(cond_list) > 1:
                for num, cond in enumerate(cond_list):
                    data_df[cond] = conditions[:, num]
                    # data_df = data_df.astype({cond: 'int32'})
            else:
                data_df[cond_list[0]] = conditions
                # data_df = data_df.astype({cond_list[0]: 'int32'})

            if stimname == "bluegreencs":
                data_df = data_df.set_index(['roi_ids', 'colors', 'locations', 'trial_nums', 'time_bin'])
                data_df_Average = data_df['preprocess_snippets'].groupby(
                    ['roi_ids', 'colors', 'locations', 'time_bin']).mean()
                data_df_Max = data_df['preprocess_snippets'].groupby(
                    ['roi_ids', 'colors', 'locations', 'trial_nums']).max()

                # calculate quality index
                single_trial_var = (
                    data_df['preprocess_snippets'].groupby(['roi_ids', 'colors', 'locations', 'trial_nums']).var())
                mean_of_var = single_trial_var.groupby(['roi_ids', 'colors', 'locations']).mean()
                var_of_mean = data_df_Average.groupby(['roi_ids', 'colors', 'locations']).var()
                quality_values = var_of_mean / mean_of_var
                max_qualities = quality_values.groupby(['roi_ids', 'colors', 'locations']).max()

                ### added by MK
            elif stimname == "movingbar":
                data_df = data_df.set_index(['roi_ids', 'directions', 'trial_nums', 'time_bin'])
                data_df_Average = data_df['preprocess_snippets'].groupby(['roi_ids', 'directions', 'time_bin']).mean()
                data_df_Max = data_df['preprocess_snippets'].groupby(['roi_ids', 'directions', 'trial_nums']).max()

                # calculate quality index
                single_trial_var = (
                    data_df['preprocess_snippets'].groupby(['roi_ids', 'directions', 'trial_nums']).var())
                mean_of_var = single_trial_var.groupby(['roi_ids', 'directions']).mean()
                var_of_mean = data_df_Average.groupby(['roi_ids', 'directions']).var()
                quality_values = var_of_mean / mean_of_var
                max_qualities = quality_values.groupby(['roi_ids', 'directions']).max()

        else:  # take average across all trials (i.e. for chirp)
            data_df = data_df.set_index(['roi_ids', 'trial_nums', 'time_bin'])
            data_df_Average = data_df['preprocess_snippets'].groupby(['roi_ids', 'time_bin']).mean()
            data_df_Max = data_df['preprocess_snippets'].groupby(['roi_ids', 'trial_nums']).max()

            # calculate quality index
            single_trial_var = (
                data_df['preprocess_snippets'].groupby(['roi_ids', 'trial_nums']).var())
            mean_of_var = single_trial_var.groupby(['roi_ids']).mean()
            var_of_mean = data_df_Average.groupby(['roi_ids']).var()
            quality_values = var_of_mean / mean_of_var
            max_qualities = quality_values  # quality_values.groupby(['roi_ids'])#.apply(pd.DataFrame)

        # determine the high quality rois
        method = (QualityParams() & key).fetch1('quality_method')
        threshold = (QualityParams() & key).fetch1('threshold')

        data_df_qi = max_qualities.reset_index()
        data_df_qi = data_df_qi.astype({'roi_ids': 'int32'})
        data_df_qi = data_df_qi.set_index(['roi_ids'])
        data_df_qi = data_df_qi.rename(columns={"preprocess_snippets": "QI"})
        datadf_qi_max = data_df_qi['QI'].groupby(['roi_ids']).max()
        if method == "percentile":
            quality_threshold = np.percentile(datadf_qi_max, threshold)
        else:
            quality_threshold = threshold
        index_ResponsivePx = np.where(datadf_qi_max >= quality_threshold)
        quality_rois = index_ResponsivePx[0] + 1
        if quality_rois.size < 5:  # check if there are any quality pixels at this threshold
            empty_quality = 1
        else:
            empty_quality = 0

        # set up data for inserting into datajoint.
        delimiter = ' '

        data_df = data_df.reset_index()
        time_bin = data_df['time_bin'].to_numpy()

        data_df_Average = data_df_Average.reset_index()
        df_Average_strs = delimiter.join(list(data_df_Average.columns))
        average_frame = data_df_Average.to_numpy()

        data_df_Max = data_df_Max.reset_index()
        df_Max_strs = delimiter.join(list(data_df_Max.columns))
        max_frame = data_df_Max.to_numpy()

        max_qualities = max_qualities.reset_index()
        max_quality_strs = delimiter.join(list(max_qualities.columns))
        quality_frame = max_qualities.to_numpy()

        self.insert1(dict(key,
                          resampled_time_bins=time_bin,
                          condition_average_frame=average_frame,
                          condition_average_strs=df_Average_strs,
                          condition_max_frame=max_frame,
                          condition_max_strs=df_Max_strs,
                          quality_indexes_frame=quality_frame,
                          quality_indexes_strs=max_quality_strs,
                          empty_quality_flag=empty_quality)
                     )

        primary_key = deepcopy(key)
        primary_key["quality_id"] = (QualityParams() & key).fetch1('quality_id')
        SummarySnippets.HighQualityPixels().insert1(dict(primary_key,
                                                         high_quality_pixels=quality_rois))
        print("populated for experimenter {}, date {}, exp {}, field {}"
              .format(key["experimenter"], key["date"], key["exp_num"],
                      key["field_id"]))


@schema
class NoiseStimulusBGFlicker(dj.Computed):
    definition = """
        # noise stimulus array, original array and upsampled to line precision for blue green flicker stim with test sequences
        ->Stimulus
        ->PreprocessTraces
        ---
        noise_array             :longblob           # array of the noise stimulus "original" (xy, frame)
        noise_array_line        :longblob           # array of the noise stimulus at line precision (x, y , line)
        """

    @property
    def key_source(self):
        rel = super().key_source
        return rel * (Stimulus & 'stimulusname in ("bluegreenflicker")').proj()

    def _make_tuples(self, key):
        # get parameters of the stimulus

        stim_params = (Stimulus.BlueGreenFlicker() & key).fetch1()
        noise_file_c = stim_params['fname_noise_cblue'] + '.txt'
        noise_file_r = stim_params['fname_noise_rblue'] + '.txt'
        noise_file_s = stim_params['fname_noise_sblue'] + '.txt'
        noise_file_g_c = stim_params['fname_noise_cgreen'] + '.txt'
        noise_file_g_r = stim_params['fname_noise_rgreen'] + '.txt'
        noise_file_g_s = stim_params['fname_noise_sgreen'] + '.txt'
        test_file_c = stim_params['fname_noise_ctest'] + '.txt'
        test_file_r = stim_params['fname_noise_rtest'] + '.txt'
        test_file_s = stim_params['fname_noise_stest'] + '.txt'

        stimulus_freq = np.int(1 / (stim_params['dur_stim_s']))

        path = (Stimulus() & key).fetch1('stim_path')
        noise_path = path + 'Blue_green_flicker_text_files/'
        noise_array_c = np.loadtxt(noise_path + noise_file_c, dtype=int, delimiter=',', skiprows=1)
        noise_array_r = np.loadtxt(noise_path + noise_file_r, dtype=int, delimiter=',', skiprows=1)
        noise_array_s = np.loadtxt(noise_path + noise_file_s, dtype=int, delimiter=',', skiprows=1)
        noise_array_g_c = np.loadtxt(noise_path + noise_file_g_c, dtype=int, delimiter=',', skiprows=1)
        noise_array_g_r = np.loadtxt(noise_path + noise_file_g_r, dtype=int, delimiter=',', skiprows=1)
        noise_array_g_s = np.loadtxt(noise_path + noise_file_g_s, dtype=int, delimiter=',', skiprows=1)

        empty_color = np.zeros(noise_array_c.shape)
        noise_array_uv = np.stack((noise_array_c, noise_array_r, noise_array_s,
                                   empty_color, empty_color, empty_color), axis=-1).T

        empty_color = np.zeros(noise_array_g_c.shape)
        noise_array_green = np.stack((empty_color, empty_color, empty_color,
                                      noise_array_g_c, noise_array_g_r, noise_array_g_s), axis=-1).T

        # get test sequences
        test_sequence_c = np.loadtxt(noise_path + test_file_c, dtype=int, delimiter=',', skiprows=1)
        test_sequence_r = np.loadtxt(noise_path + test_file_r, dtype=int, delimiter=',', skiprows=1)
        test_sequence_s = np.loadtxt(noise_path + test_file_s, dtype=int, delimiter=',', skiprows=1)
        empty_color = np.zeros(test_sequence_c.shape)

        test_seq_uv = np.stack((test_sequence_c, test_sequence_r, test_sequence_s,
                                empty_color, empty_color, empty_color), axis=-1).T
        test_seq_green = np.stack((empty_color, empty_color, empty_color,
                                   test_sequence_c, test_sequence_r, test_sequence_s), axis=-1).T

        # recreate all frames of the stimulus
        stim_order_string = stim_params['order_string']
        stim_sequence_array = np.array(stim_order_string.split('_'))
        noise_array_updated = np.array([], dtype=np.int64).reshape(6, 0)

        for stim_type in stim_sequence_array:

            if 'uv' in stim_type:
                epoch_number = int(stim_type[-1]) - 1
                epoch_start_line = stim_params['epoch_start_lines'][epoch_number]
                epoch_end_line = stim_params['epoch_end_lines'][epoch_number]
                current_portion = noise_array_uv[:, epoch_start_line:epoch_end_line]
                noise_array_updated = np.hstack([noise_array_updated, current_portion])

            elif 'green' in stim_type:
                epoch_number = int(stim_type[-1]) - 1
                epoch_start_line = stim_params['epoch_start_lines'][epoch_number]
                epoch_end_line = stim_params['epoch_end_lines'][epoch_number]
                current_portion = noise_array_green[:, epoch_start_line:epoch_end_line]
                noise_array_updated = np.hstack([noise_array_updated, current_portion])

            elif stim_type == 'test':
                noise_array_updated = np.hstack([noise_array_updated, test_seq_uv, test_seq_green])

            elif stim_type == 'black':
                print('warning:skipping black sequences during noise stimulus array creation')
                print('for details see schema documentation')
                # A note here: for black sequences that are not a multiple of the stimulus frame rate,
                # we can't add to noise array here. These will be inserted at line precision

                # could update this to test for the break being a multiple of the stim frame rate.
            else:
                print("unrecognized stimulus sequence")

        # get information for upsampling to line precision
        triggertimes = (TraceTriggerTimes() & key).fetch1('trigger_times_corrected')  # triggertimes in seconds
        nY = (Presentation() & key).fetch1('scan_num_lines')
        n_triggers = triggertimes.shape[0]

        # # add the stimulutor delay to trigger times
        # path = (Experiment() & key).fetch1('headerpath')
        # path = path + 'Pre/'
        # file = (Field.RoiMask() & key).fetch1('fromfile')
        # datafile = h5py.File(path + file, 'r')
        # os_parameters = np.copy(datafile['OS_Parameters'])
        # stimulator_delay = os_parameters[8] / 1000  # converting from ms to seconds
        # datafile.close()
        #
        # # add the stimulator delay which is also in seconds
        # triggertimes = triggertimes + stimulator_delay

        # upsample to line precision
        line_duration = (Presentation() & key).fetch1('scan_line_duration')
        lines_per_second = 1 / line_duration
        triggertimes_line = np.round(triggertimes / line_duration)  # to convert seconds to lines

        # make stimulus
        traces = (PreprocessTraces() & key).fetch1('preprocess_traces')
        n_frames = traces.shape[0]

        t_between_trigs = np.diff(triggertimes)

        noise_array_line = np.zeros((6, np.int(n_frames * nY)))
        current_frame = 0
        for tt in range(n_triggers - 1):
            # print(tt)
            if np.around(t_between_trigs[tt], decimals=1) == stim_params['dur_break']:  # break period left as zeros
                print("leaving break periods as zeros for trigger #", tt)
                print("trigger time of break in lines is ", triggertimes_line[tt])
            else:
                loop_range = np.int(
                    np.round(stimulus_freq * t_between_trigs[tt]))  # figure out if main epoch or test sequence

                if loop_range > stimulus_freq:
                    print("test_trigger at ", triggertimes_line[tt])
                    # print("loop range is", loop_range)
                    for nn in range(loop_range):
                        timestart = np.int(
                            triggertimes_line[tt] + (nn / stimulus_freq) * lines_per_second)
                        timestop = np.int(triggertimes_line[tt] + ((nn + 1) / stimulus_freq) * lines_per_second)
                        noise_array_line[:, timestart:timestop] = np.tile(noise_array_updated[:, current_frame],
                                                                          (timestop - timestart, 1)).T
                        current_frame += 1
                else:
                    for nn in range(loop_range):
                        timestart = np.int(
                            triggertimes_line[tt] + (nn / stimulus_freq) * (
                                    triggertimes_line[tt + 1] - triggertimes_line[tt]))
                        timestop = np.int(triggertimes_line[tt] + ((nn + 1) / stimulus_freq) * (
                                triggertimes_line[tt + 1] - triggertimes_line[tt]))
                        noise_array_line[:, timestart:timestop] = np.tile(noise_array_updated[:, current_frame],
                                                                          (timestop - timestart, 1)).T
                        current_frame += 1

        self.insert1(dict(key,
                          noise_array=noise_array_updated,
                          noise_array_line=noise_array_line))


@schema
class BC_NoiseStimulus(dj.Computed):
    definition = """
    # noise stimulus array, original array and upsampled to line precision
    ->Stimulus
    ->PreprocessTraces
    ---
    noise_array             :longblob           # array of the noise stimulus "original" (xy, frame)
    noise_array_line        :longblob           # array of the noise stimulus at line precision (x, y , line)
    """

    @property
    def key_source(self):
        rel = super().key_source
        return rel * (Stimulus & 'stimulusname in ("blue_csflicker", "green_csflicker")').proj()

    def _make_tuples(self, key):
        # get parameters of the stimulus
        stimulus_name = (Stimulus() & key).fetch1('stimulusname')
        path = (Stimulus() & key).fetch1('stim_path')
        noise_path = path + 'BCs_Color_Szatko2020/'
        # get stimulus params and open file with the stimulus
        if stimulus_name == 'blue_csflicker':
            stim_params = (Stimulus.Blue_CSFlicker() & key).fetch1()
            noise_file_c = stim_params['fname_noise_cblue'] + '.txt'
            noise_file_s = stim_params['fname_noise_sblue'] + '.txt'
            noise_array_c = np.loadtxt(noise_path + noise_file_c, dtype=int, delimiter=',', skiprows=1)[:,0]
            noise_array_s = np.loadtxt(noise_path + noise_file_s, dtype=int, delimiter=',', skiprows=1)[:,1]
            noise_array = np.stack((noise_array_c, noise_array_s), axis=-1).T

        elif stimulus_name == 'green_csflicker':
            stim_params = (Stimulus.Green_CSFlicker() & key).fetch1()
            noise_file_c = stim_params['fname_noise_cgreen'] + '.txt'
            noise_file_s = stim_params['fname_noise_sgreen'] + '.txt'
            noise_array_c = np.loadtxt(noise_path + noise_file_c, dtype=int, delimiter=',', skiprows=1)[:,2]
            noise_array_s = np.loadtxt(noise_path + noise_file_s, dtype=int, delimiter=',', skiprows=1)[:,3]
            noise_array = np.stack((noise_array_c, noise_array_s), axis=-1).T

        stimulus_freq = np.int(1 / (stim_params['dur_stim_s']))

        # get information for upsampling to line precision
        triggertimes = (TraceTriggerTimes() & key).fetch1('trigger_times_corrected')  # triggertimes in seconds
        nY = (Presentation() & key).fetch1('scan_num_lines')
        n_triggers = triggertimes.shape[0]

        # upsample to line precision
        line_duration = (Presentation() & key).fetch1('scan_line_duration')
        lines_per_second = 1 / line_duration
        triggertimes_line = np.round(triggertimes / line_duration)  # to convert seconds to lines

        # make stimulus
        traces = (PreprocessTraces() & key).fetch1('preprocess_traces')
        n_frames = traces.shape[0]

        if stimulus_name == 'blue_csflicker' or stimulus_name == 'green_csflicker':
            noise_array_line = np.zeros((2, np.int(n_frames * nY)))

            for tt in range(n_triggers - 1):
                for nn in range(stimulus_freq):
                    timestart = np.int(
                        triggertimes_line[tt] + (nn / stimulus_freq) * (
                                triggertimes_line[tt + 1] - triggertimes_line[tt]))
                    timestop = np.int(triggertimes_line[tt] + ((nn + 1) / stimulus_freq) * (
                            triggertimes_line[tt + 1] - triggertimes_line[tt]))
                    noise_array_line[0, timestart:timestop] = np.tile(noise_array_c[tt * stimulus_freq + nn],
                                                                      (timestop - timestart, 1)).T
                    noise_array_line[1, timestart:timestop] = np.tile(noise_array_s[tt * stimulus_freq + nn],
                                                                      (timestop - timestart, 1)).T
            self.insert1(dict(key,
                              noise_array=noise_array,
                              noise_array_line=noise_array_line))


@schema
class BGFlickerParams(dj.Lookup):
    definition = """
    # Parameters for getting noise kernels
    noise_id            :int            # arbitrary integer identifier
    ---
    filter_length       :float           # time in seconds for the filter
    """


@schema
class InterpolatedTraces(dj.Computed):
    definition = """
    ->Field
    ->PreprocessTraces
    ->NoiseStimulusBGFlicker
    ---
    interpolated_traces         :longblob           # traces obtained from linear interpolation (roi x time)
    """

    def _make_tuples(self, key):
        # get trace data
        line_duration = (Presentation() & key).fetch1('scan_line_duration')
        traces = (PreprocessTraces() & key).fetch1('preprocess_traces')
        traces_times_all = (TraceTriggerTimes() & key).fetch1('trace_times')

        # get the stimulus (already at line precision)
        stimulus = (NoiseStimulusBGFlicker() & key).fetch1('noise_array_line')

        # do linear interpolation of the traces
        # set up time array
        tstart = 0
        nlines = stimulus.shape[1]
        tend = line_duration * nlines
        timeALL = np.linspace(tstart, tend, nlines)
        # make arrays for linear interpolation
        yold = traces.T
        xold = traces_times_all.T
        nrowsold = yold.shape[0]  # number of rows in the x_t and y traces
        timenew = np.tile(timeALL, (nrowsold, 1))
        # the interpolation function
        arraytuples = [*zip(timenew, xold, yold)]
        pool = Pool(os.cpu_count())  # use multiple nodes to iterate over the data
        result = (pool.starmap(np.interp, iterable=arraytuples))  # result contains the interpolated y values
        # result is list of arrays with each item representing a roi
        ynew = np.array(result)
        self.insert1(dict(key,
                          interpolated_traces=ynew,
                          ))


@schema
class BlueGreenFlickerKernels2(dj.Computed):
    definition = """
    # Blue and Green Flicker Kernels calculated using matrix multiplication
    ->Field
    ->InterpolatedTraces
    ->NoiseStimulusBGFlicker
    ->BGFlickerParams
    ---
    st_kernels_uv                   :longblob         # space-time kernels (space, line, roi) for uv epochs
    st_kernels_green                :longblob         # space-time kernels (space, line, roi) for green epochs
    offset_before                   :int              # line offset used in kernel and convolution calculations
    offset_after                    :int              # line offset used in kernel and convolution calculations
    kernel_length_line              :int              # kernel length in lines
    first_uv_stim                   :blob             # array of the first UV stimulus epoch
    second_uv_stim                  :blob             # array of the second UV stimulus epoch
    first_green_stim                :blob             # array of the first green stimulus epoch
    second_green_stim               :blob             # array of the second green stimulus epoch
    uv_trigger_starts_lines         :blob             # lines where the UV epochs start
    uv_trigger_ends_lines           :blob             # lines where the UV epochs end
    green_trigger_starts_lines      :blob             # lines where the green epochs start
    green_trigger_ends_lines        :blob             # lines where the green epochs end
    """

    def _make_tuples(self, key):

        # get trace data
        line_duration = (Presentation() & key).fetch1('scan_line_duration')
        traces_times_all = (TraceTriggerTimes() & key).fetch1('trace_times')
        ynew = np.array((InterpolatedTraces & key).fetch1('interpolated_traces'))
        # get the stimulus (already at line precision)
        stimulus = (NoiseStimulusBGFlicker() & key).fetch1('noise_array_line')
        triggertimes = (TraceTriggerTimes() & key).fetch1('trigger_times_corrected')

        # # add the stimulutor delay to trigger times
        # path = (Experiment() & key).fetch1('headerpath')
        # path = path + 'Pre/'
        # file = (Field.RoiMask() & key).fetch1('fromfile')
        # datafile = h5py.File(path + file, 'r')
        # os_parameters = np.copy(datafile['OS_Parameters'])
        # stimulator_delay = os_parameters[8] / 1000
        # datafile.close()
        # triggertimes = triggertimes + stimulator_delay

        # get triggers of the main epochs for blue and green
        stim_params = (Stimulus.BlueGreenFlicker() & key).fetch1()
        stim_order_string = stim_params['order_string']
        stim_sequence_array = np.array(stim_order_string.split('_'))
        main_starts = np.array(stim_params['epoch_start_lines'])
        main_ends = np.array(stim_params['epoch_end_lines'])
        main_epoch_length_stim_f = main_ends[0] - main_starts[0]
        n_triggers_main_epoch = np.int(
            main_epoch_length_stim_f * stim_params['dur_stim_s'] / stim_params['trigger_freq'])
        main_epoch_array_uv = np.zeros(n_triggers_main_epoch)
        main_epoch_array_uv[0] = 1
        main_epoch_array_uv[-1] = 2
        main_epoch_array_green = np.zeros(n_triggers_main_epoch)
        main_epoch_array_green[0] = 3
        main_epoch_array_green[-1] = 4
        test_epoch_array = np.zeros(len(stim_params['test_seq_triggers']) * 2)
        black_epoch_array = np.zeros(1)
        trigger_sequence = np.array([], dtype=np.int64).reshape(0)

        # make an array that marks the triggers for the UV and green epochs
        for stim_type in stim_sequence_array:
            if 'uv' in stim_type:
                trigger_sequence = np.hstack([trigger_sequence, main_epoch_array_uv])
            elif 'green' in stim_type:
                trigger_sequence = np.hstack([trigger_sequence, main_epoch_array_green])
            elif stim_type == 'test':
                trigger_sequence = np.hstack([trigger_sequence, test_epoch_array])
            elif stim_type == 'black':
                trigger_sequence = np.hstack([trigger_sequence, black_epoch_array])
            else:
                print("unrecognized stimulus sequence")

        uv_trigger_starts_lines = np.floor(triggertimes[trigger_sequence == 1] / line_duration)
        uv_trigger_ends_lines = np.floor(triggertimes[trigger_sequence == 2] / line_duration)
        green_trigger_starts_lines = np.floor(triggertimes[trigger_sequence == 3] / line_duration)
        green_trigger_ends_lines = np.floor(triggertimes[trigger_sequence == 4] / line_duration)

        # # get the line offset for each ROI
        # line_offsets = np.floor(traces_times_all[0, :] / line_duration).astype(
        #     'int32')

        # get the traces to multiply
        first_uv_traces = np.zeros(
            (ynew.shape[0], np.int(uv_trigger_ends_lines[0]) - np.int(uv_trigger_starts_lines[0])))
        second_uv_traces = np.zeros(
            (ynew.shape[0], np.int(uv_trigger_ends_lines[1]) - np.int(uv_trigger_starts_lines[1])))
        first_green_traces = np.zeros(
            (ynew.shape[0], np.int(green_trigger_ends_lines[0]) - np.int(green_trigger_starts_lines[0])))
        second_green_traces = np.zeros(
            (ynew.shape[0], np.int(green_trigger_ends_lines[1]) - np.int(green_trigger_starts_lines[1])))
        for i in range(ynew.shape[0]):
            first_uv_traces[i, :] = ynew[i, np.int(uv_trigger_starts_lines[0]):np.int(
                uv_trigger_ends_lines[0])]
            second_uv_traces[i, :] = ynew[i, np.int(uv_trigger_starts_lines[1]):np.int(
                uv_trigger_ends_lines[1])]
            first_green_traces[i, :] = ynew[i, np.int(green_trigger_starts_lines[0]):np.int(
                green_trigger_ends_lines[0])]
            second_green_traces[i, :] = ynew[i, np.int(green_trigger_starts_lines[1]):np.int(
                green_trigger_ends_lines[1])]

        # crop stimuli to only include the UV and green stimulated periods
        # also cropping the stimuli to just include one color channel since they were not played simultaneously
        first_uv_stim = stimulus[:3, np.int(uv_trigger_starts_lines[0]):np.int(uv_trigger_ends_lines[0])]
        second_uv_stim = stimulus[:3, np.int(uv_trigger_starts_lines[1]):np.int(uv_trigger_ends_lines[1])]
        first_green_stim = stimulus[3:, np.int(green_trigger_starts_lines[0]):np.int(green_trigger_ends_lines[0])]
        second_green_stim = stimulus[3:, np.int(green_trigger_starts_lines[1]):np.int(green_trigger_ends_lines[1])]

        # choose the kernel length here
        kernel_params = (BGFlickerParams & key).fetch1()
        kernel_length_s = kernel_params['filter_length']
        kernel_length_line = np.int(np.floor(kernel_length_s / line_duration))
        offset_after = np.int(
            np.floor(kernel_length_line * .25))  # lines to include into the future (using 1/4 of kernel length)
        offset_before = kernel_length_line - offset_after

        # get the UV kernels
        kernels_all_rois_uv = self.matrix_multiply(first_uv_traces, first_uv_stim, second_uv_traces, second_uv_stim,
                                                   offset_before, offset_after, kernel_length_line)
        # get the green kernels
        kernels_all_rois_green = self.matrix_multiply(first_green_traces, first_green_stim, second_green_traces,
                                                      second_green_stim,
                                                      offset_before, offset_after, kernel_length_line)

        self.insert1(dict(key,
                          st_kernels_uv=kernels_all_rois_uv,
                          st_kernels_green=kernels_all_rois_green,
                          offset_before=offset_before,
                          offset_after=offset_after,
                          kernel_length_line=kernel_length_line,
                          first_uv_stim=first_uv_stim,
                          second_uv_stim=second_uv_stim,
                          first_green_stim=first_green_stim,
                          second_green_stim=second_green_stim,
                          # line_offsets=line_offsets,
                          uv_trigger_starts_lines=uv_trigger_starts_lines,
                          uv_trigger_ends_lines=uv_trigger_ends_lines,
                          green_trigger_starts_lines=green_trigger_starts_lines,
                          green_trigger_ends_lines=green_trigger_ends_lines,
                          ))

    def matrix_multiply(self, first_traces, first_stim, second_traces, second_stim, offset_before, offset_after,
                       kernel_length_line):  # get the kernels
        first_traces_cropped = first_traces[:, offset_before:-offset_after]
        second_traces_cropped = second_traces[:, offset_before:-offset_after]
        design_matrix1 = np.zeros((kernel_length_line, first_traces_cropped.shape[1], first_stim.shape[0]))
        trace_length = first_traces_cropped.shape[1]
        for i in range(kernel_length_line):
            design_matrix1[i, :, :] = first_stim[:, i:trace_length + i].T
        design_matrix2 = np.zeros((kernel_length_line, second_traces_cropped.shape[1], second_stim.shape[0]))
        trace_length = second_traces_cropped.shape[1]
        for i in range(kernel_length_line):
            design_matrix2[i, :, :] = second_stim[:, i:trace_length + i].T
        # concatenate the first and second parts of the traces and stimulus matrices
        full_matrix = np.concatenate((design_matrix1, design_matrix2), axis=1)
        full_traces = np.concatenate((first_traces_cropped, second_traces_cropped), axis=1)
        # Z scoring traces and stimulus matrix
        for i in range(3):
            full_matrix[:, :, i] = (full_matrix[:, :, i].T - np.mean(full_matrix[:, :, i], axis=1)).T
            full_matrix[:, :, i] = (full_matrix[:, :, i].T / np.std(full_matrix[:, :, i], axis=1)).T
        full_traces = (full_traces.T - np.mean(full_traces, axis=1)).T
        full_traces = (full_traces.T / np.std(full_traces, axis=1)).T
        kernels_all_rois = np.matmul(full_traces, full_matrix)
        kernels_all_rois /=full_traces.shape[1] #SS this is new
        return kernels_all_rois


@schema
class ArtefactKernel(dj.Manual):
    definition = """
    # Artefact kernel to use for correcting the UV artefact in traces and kernels

    artefact_kernel_id          :tinyint            # Unique integer identifier
    ---
    artefact_kernel             :blob           # contains the artefact kernel for the UV surround
    """


@schema
class CorrectedTraces(dj.Computed):
    definition = """
    # Traces corrected by subtracting the stimulus convolved with the weighted artefact
    ->ArtefactKernel
    ->BlueGreenFlickerKernels2
    ---
    corrected_traces            :longblob           # contains the corrected traces by roi
    alpha                       :blob               # contains the alpha by roi
    """

    def _make_tuples(self, key):
        ynew = np.array((InterpolatedTraces & key).fetch1('interpolated_traces'))
        artefact_kernel = (ArtefactKernel & key).fetch1('artefact_kernel')
        stimulus = (NoiseStimulusBGFlicker() & key).fetch1('noise_array_line')
        offset_before = (BlueGreenFlickerKernels2 & key).fetch1('offset_before')
        offset_after = (BlueGreenFlickerKernels2 & key).fetch1('offset_after')

        # Estimate artefact trace by convoling artefact kernel with UV surround stimulus
        kernel_past = offset_before - 1
        kernel_future = offset_after
        trace_artifact = np.convolve(stimulus[2, :], np.flip(artefact_kernel), mode='full')[kernel_future:-kernel_past]

        # Before subracting the artefact trace from the traces we need to estimate a weight alpha for each roi
        # Then artefact trace weighted by alpha gets subtracted from trace
        alpha = np.zeros(ynew.shape[0])
        corrected_traces = np.zeros_like(ynew)
        denominator = np.matmul(trace_artifact, trace_artifact)

        for roi in range(ynew.shape[0]):
            current_alpha = np.amax((0, np.matmul(trace_artifact, ynew[roi, :]) / denominator))
            alpha[roi] = current_alpha
            corrected_traces[roi, :] = ynew[roi, :] - (current_alpha * trace_artifact)

        self.insert1(dict(key,
                          corrected_traces=corrected_traces,
                          alpha=alpha))


@schema
class CorrectedBGFlickerKernels(dj.Computed):
    definition = """
        # Blue and Green Flicker Kernels calculated using matrix multiplication after artefact correction
        ->Field
        ->CorrectedTraces
        ---
        st_kernels_uv_corrected              :longblob         # space-time kernels (space, line, roi) for uv epochs
        st_kernels_green_corrected           :longblob         # space-time kernels (space, line, roi) for green epochs
        """

    def _make_tuples(self, key):

        # get trace data
        ynew = np.array((CorrectedTraces & key).fetch1('corrected_traces'))

        # import stuff from previous run of kernel estimation on the uncorrected traces
        kernel_calc_inputs = (BlueGreenFlickerKernels2 & key).fetch1()
        kernel_length_line = kernel_calc_inputs['kernel_length_line']
        first_uv_stim = kernel_calc_inputs['first_uv_stim']
        second_uv_stim = kernel_calc_inputs['second_uv_stim']
        first_green_stim = kernel_calc_inputs['first_green_stim']
        second_green_stim = kernel_calc_inputs['second_green_stim']
        offset_before = kernel_calc_inputs['offset_before']
        offset_after = kernel_calc_inputs['offset_after']
        uv_trigger_starts_lines = kernel_calc_inputs['uv_trigger_starts_lines']
        uv_trigger_ends_lines = kernel_calc_inputs['uv_trigger_ends_lines']
        green_trigger_starts_lines = kernel_calc_inputs['green_trigger_starts_lines']
        green_trigger_ends_lines = kernel_calc_inputs['green_trigger_ends_lines']
        # line_offsets = kernel_calc_inputs['line_offsets']

        # get the traces to multiply
        # here the traces for each roi are realigned to their line offsets
        first_uv_traces = np.zeros(
            (ynew.shape[0], np.int(uv_trigger_ends_lines[0]) - np.int(uv_trigger_starts_lines[0])))
        second_uv_traces = np.zeros(
            (ynew.shape[0], np.int(uv_trigger_ends_lines[1]) - np.int(uv_trigger_starts_lines[1])))
        first_green_traces = np.zeros(
            (ynew.shape[0], np.int(green_trigger_ends_lines[0]) - np.int(green_trigger_starts_lines[0])))
        second_green_traces = np.zeros(
            (ynew.shape[0], np.int(green_trigger_ends_lines[1]) - np.int(green_trigger_starts_lines[1])))
        for i in range(ynew.shape[0]):
            first_uv_traces[i, :] = ynew[i, np.int(uv_trigger_starts_lines[0]):np.int(
                uv_trigger_ends_lines[0])]
            second_uv_traces[i, :] = ynew[i, np.int(uv_trigger_starts_lines[1]):np.int(
                uv_trigger_ends_lines[1])]
            first_green_traces[i, :] = ynew[i, np.int(green_trigger_starts_lines[0]):np.int(
                green_trigger_ends_lines[0])]
            second_green_traces[i, :] = ynew[i, np.int(green_trigger_starts_lines[1]):np.int(
                green_trigger_ends_lines[1])]

        # get the UV kernels
        kernels_all_rois_uv = self.matrix_multiply(first_uv_traces, first_uv_stim, second_uv_traces, second_uv_stim,
                                                   offset_before, offset_after, kernel_length_line)
        # get the green kernels
        kernels_all_rois_green = self.matrix_multiply(first_green_traces, first_green_stim, second_green_traces,
                                                      second_green_stim,
                                                      offset_before, offset_after, kernel_length_line)

        self.insert1(dict(key,
                          st_kernels_uv_corrected=kernels_all_rois_uv,
                          st_kernels_green_corrected=kernels_all_rois_green,
                          ))

    def matrix_multiply(self, first_traces, first_stim, second_traces, second_stim, offset_before, offset_after,
                       kernel_length_line):  # get the kernels
        first_traces_cropped = first_traces[:, offset_before:-offset_after]
        second_traces_cropped = second_traces[:, offset_before:-offset_after]
        design_matrix1 = np.zeros((kernel_length_line, first_traces_cropped.shape[1], first_stim.shape[0]))
        trace_length = first_traces_cropped.shape[1]
        for i in range(kernel_length_line):
            design_matrix1[i, :, :] = first_stim[:, i:trace_length + i].T
        design_matrix2 = np.zeros((kernel_length_line, second_traces_cropped.shape[1], second_stim.shape[0]))
        trace_length = second_traces_cropped.shape[1]
        for i in range(kernel_length_line):
            design_matrix2[i, :, :] = second_stim[:, i:trace_length + i].T
        # concatenate the first and second parts of the traces and stimulus matrices
        full_matrix = np.concatenate((design_matrix1, design_matrix2), axis=1)
        full_traces = np.concatenate((first_traces_cropped, second_traces_cropped), axis=1)
        # Z scoring traces and stimulus matrix
        for i in range(3):
            full_matrix[:, :, i] = (full_matrix[:, :, i].T - np.mean(full_matrix[:, :, i], axis=1)).T
            full_matrix[:, :, i] = (full_matrix[:, :, i].T / np.std(full_matrix[:, :, i], axis=1)).T
        full_traces = (full_traces.T - np.mean(full_traces, axis=1)).T
        full_traces = (full_traces.T / np.std(full_traces, axis=1)).T
        kernels_all_rois = np.matmul(full_traces, full_matrix)
        kernels_all_rois /=full_traces.shape[1]
        return kernels_all_rois


@schema
class MinMaxQualityParams(dj.Lookup):
    definition = """
    # Params for calculating the quality values
    param_id            :tinyint        # unique integer identifier
    ---
    start_kernel        :int            # line at which to start the min max estimate
    """


@schema
class MinMaxQuality(dj.Computed):
    definition = """
    # Quality filtering of kernels based on the max minus min
    ->MinMaxQualityParams
    ->CorrectedBGFlickerKernels
    ---
    quality_values            :blob               # maximum amplitude in each kernel
    """
    def _make_tuples(self, key):
        # get the kernels
        kernels_uv_all_rois = (CorrectedBGFlickerKernels & key).fetch1('st_kernels_uv_corrected')
        kernels_green_all_rois = (CorrectedBGFlickerKernels & key).fetch1('st_kernels_green_corrected')

        # get parameter for kernel window
        start_kernel = (MinMaxQualityParams & key).fetch1('start_kernel')
        stop_kernel = int(kernels_green_all_rois.shape[0]*3/4) #stop at zero time

        # Calculate max and min values during response period
        min_uv = np.amin(kernels_uv_all_rois[start_kernel:stop_kernel, :, :], axis=0)
        max_uv = np.amax(kernels_uv_all_rois[start_kernel:stop_kernel, :, :], axis=0)
        min_green = np.amin(kernels_green_all_rois[start_kernel:stop_kernel, :, :], axis=0)
        max_green = np.amax(kernels_green_all_rois[start_kernel:stop_kernel, :, :], axis=0)

        # Calculate kernel amplitude
        amplitude_uv = np.abs(max_uv - min_uv)
        amplitude_green = np.abs(max_green - min_green)

        # Take largest amplitude across conditions; Do *not* include UV surround as it has the artefact
        max_amplitude = np.amax(np.concatenate((amplitude_uv[:, 0:2], amplitude_green), axis=1), axis=1)

        self.insert1(dict(key,
                          quality_values=max_amplitude,
                          ))


@schema
class BGFullResponses(dj.Computed):
    definition = """
    # full responses to the blue green flicker stimulus to use for training the model
    ->CorrectedTraces
    ->NoiseStimulusBGFlicker
    ---
    full_response_traces        :longblob              # responses to the entire stimulated period (with unstimulated baseline periods removed)
    full_stimulus               :longblob              # the full stimulus
    """

    def _make_tuples(self, key):

        #get the corrected traces
        ynew_corrected = np.array((CorrectedTraces() & key).fetch1('corrected_traces'))

        #get the line precision stimulus
        stimulus = (NoiseStimulusBGFlicker() & key).fetch1('noise_array_line')

        #get the triggers
        triggertimes = (TraceTriggerTimes() & key).fetch1('trigger_times_corrected')
        line_duration = (Presentation() & key).fetch1('scan_line_duration')

        #extract the stimulated period
        triggertimes_line = triggertimes/line_duration

        start_line = np.int(np.round(triggertimes_line[0]))
        # end_line = np.int(np.round(triggertimes_line[-1])) #there is a trigger at the end of the stimulus
        end_line = start_line + 346266 #this is a hardcoded value to make all of these arrays the same length

        full_response_traces = ynew_corrected[:,start_line:end_line]
        full_stimulus = stimulus[:,start_line:end_line]

        self.insert1(dict(key,
                          full_response_traces=full_response_traces,
                          full_stimulus=full_stimulus,
                          ))


@schema
class BGTestSequences(dj.Computed):
    definition = """
    ->CorrectedTraces
    ->NoiseStimulusBGFlicker
    ---
    test_sequences             :longblob               # responses to test sequences roi x lines x repetition
    test_sequences_stims       :longblob               # stimulus at line precision spatial color channel x lines x repetition
    """

    class BGQualityIndex(dj.Part):
        definition = """
        ->BGTestSequences
        ---
        quality_values          :blob           #array of quality values by roi
        """

    def _make_tuples(self, key):

        #get the corrected traces
        ynew_corrected = np.array((CorrectedTraces() & key).fetch1('corrected_traces'))

        #get the line precision stimulus
        stimulus = (NoiseStimulusBGFlicker() & key).fetch1('noise_array_line')

        # get triggers of the main epochs for blue and green
        stim_params = (Stimulus.BlueGreenFlicker() & key).fetch1()
        stim_order_string = stim_params['order_string']
        stim_sequence_array = np.array(stim_order_string.split('_'))
        main_starts = np.array(stim_params['epoch_start_lines'])
        main_ends = np.array(stim_params['epoch_end_lines'])
        main_epoch_length_stim_f = main_ends[0] - main_starts[0]
        n_triggers_main_epoch = np.int(
            main_epoch_length_stim_f * stim_params['dur_stim_s'] / stim_params['trigger_freq'])
        main_epoch_array_uv = np.zeros(n_triggers_main_epoch)
        main_epoch_array_green = np.zeros(n_triggers_main_epoch)
        test_epoch_array = np.zeros(len(stim_params['test_seq_triggers']) * 2)
        test_epoch_array[0] = 1
        test_epoch_array[-1] = 2
        black_epoch_array = np.zeros(1)
        trigger_sequence = np.array([], dtype=np.int64).reshape(0)

        # make an array that marks the triggers for the UV and green epochs
        for stim_type in stim_sequence_array:
            if 'uv' in stim_type:
                trigger_sequence = np.hstack([trigger_sequence, main_epoch_array_uv])
            elif 'green' in stim_type:
                trigger_sequence = np.hstack([trigger_sequence, main_epoch_array_green])
            elif stim_type == 'test':
                trigger_sequence = np.hstack([trigger_sequence, test_epoch_array])
            elif stim_type == 'black':
                trigger_sequence = np.hstack([trigger_sequence, black_epoch_array])
            else:
                print("unrecognized stimulus sequence")

        #get the trigger times
        triggertimes = (TraceTriggerTimes() & key).fetch1('trigger_times_corrected')
        line_duration = (Presentation() & key).fetch1('scan_line_duration')

        test_starts_lines = np.round(triggertimes[trigger_sequence == 1] / line_duration)
        test_ends_lines = np.round(triggertimes[trigger_sequence == 2] / line_duration)

        #this adds some baseline time to beginning and end of test sequences
        test_starts_lines2 = np.round((triggertimes[
                            trigger_sequence == 1] - 2) / line_duration)  # start it 2 seconds early to see baseline/black period
        test_ends_lines2 = np.round((triggertimes[
            trigger_sequence == 2] + 5) / line_duration) # adding 5 s to get to end of the epoch

        #get the test sequences from the corrected traces
        #roi x lines x repetition
        test_sequences = np.zeros(
                    (ynew_corrected.shape[0], np.int(np.amin(test_ends_lines2 - test_starts_lines2)), test_starts_lines2.shape[0]))

        #spatial/color channel x lines x repetition
        test_sequences_stims = np.zeros((stimulus.shape[0], np.int(np.amin(test_ends_lines2 - test_starts_lines2)), test_starts_lines2.shape[0]))

        #correct potential rounding errors
        test_lengths = test_ends_lines2 - test_starts_lines2
        for i in range(test_lengths.shape[0]):
            if test_lengths[i] > test_sequences.shape[1]:
                test_ends_lines2[i] = test_ends_lines2[i] - 1

        for i in range(test_starts_lines.shape[0]):
            test_start = test_starts_lines[i]
            test_end = test_ends_lines[i]

            test_start = test_starts_lines2[i]
            test_end = test_ends_lines2[i]

            test_sequences[:,:,i] = ynew_corrected[:, np.int(test_start):np.int(test_end)]

            test_sequences_stims[:,:,i] = stimulus[:, np.int(test_start):np.int(test_end)]

        # calculate quality indices
        single_trial_variance = np.var(test_sequences, axis=1)
        mean_of_trial_var = np.mean(single_trial_variance, axis=1)
        mean_traces = np.mean(test_sequences, axis=2)
        var_of_mean = np.var(mean_traces, axis=1)
        qi = var_of_mean / mean_of_trial_var

        self.insert1(dict(key,
                          test_sequences=test_sequences,
                          test_sequences_stims=test_sequences_stims,
                          ))

        BGTestSequences.BGQualityIndex().insert1(dict(key,
                                                      quality_values=qi,
                                                      ))


@schema
class IplBorders(dj.Manual):
    definition = """
    # Manually determined index and thickness information for the IPL. See XZ widget notebook to determine values
    ->Field
    ---
    left            :tinyint            #pixel index where gcl/ipl border intersects on left of image (with GCL up)
    right           :tinyint            #pixel index where gcl/ipl border intersects on right side of image
    thick           :tinyint            #pixel width of the ipl
    """


@schema
class IplDepth(dj.Computed):
    definition = """
        # Gives the IPL depth of the field's ROIs relative to the GCL (=0) and INL (=1)
        ->IplBorders
        ->Presentation
        ---
        geoc          :blob           #Geometric positions of ROIs in terms of pixels, roi x position
        depth         :blob           #Depth in the IPL ordered by ROI
        """

    def _make_tuples(self, key):
        left, right, thick = (IplBorders() & key).fetch1('left', 'right', 'thick')
        roimask = (Field.RoiMask() & key).fetch1('roi_mask')
        x_zoom = (Presentation.ScanInfo() & key).fetch1('zoom')
        z_zoom = (Presentation.ScanInfo() & key).fetch1('user_zoomz')
        date = key["date"]
        date = date.strftime('%Y-%m-%d')

        if "2021" not in date:
            um_pixel_x = (71.5 / roimask.shape[0]) / x_zoom
            um_pixel_z = (61.5 / roimask.shape[1]) / z_zoom
        else:
            um_pixel_x = (65 / roimask.shape[0]) / x_zoom
            um_pixel_z = (61 / roimask.shape[1]) / z_zoom

        # get the Geometric centers of the rois calculated from Igor
        path = (Experiment() & key).fetch1('headerpath')
        path = path + 'Pre/'
        file = (Field.RoiMask() & key).fetch1('fromfile')
        datafile = h5py.File(path + file, 'r')
        geoc = datafile['GeoC']
        geoc = np.copy(geoc)
        datafile.close()

        # geoc_pixels_corrected = geoc / um_pixel_x #AV
        # traces = (PreprocessTraces() & key).fetch1('preprocess_traces')
        # n_rois = traces.shape[1]
        # geoc_pixels_corrected = np.zeros((n_rois, 2))
        # geoc_pixels_corrected[:, 0] = geoc[:, 0] / um_pixel_x
        # geoc_pixels_corrected[:, 1] = geoc[:, 1] / um_pixel_z

        # calculate the depth relative to the IPL borders
        m1, b1 = self.get_line([(0, left), (roimask.shape[0] - 1, right)])
        shifts = m1 * geoc[:, 0] + b1  # geoc_pixels_corrected[:, 0]
        depth = (geoc[:, 1] - shifts) / thick  # geoc_pixels_corrected[:, 1]

        self.insert1(dict(key,
                          geoc=geoc,  # geoc_pixels_corrected
                          depth=depth,
                          ))

    def get_line(self, points):
        x_coords, y_coords = zip(*points)
        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
        m, c = lstsq(A, y_coords, rcond=None)[0]
        return m, c


@schema
class RelativeFieldLocation(dj.Computed):
    definition = """
        #location of the recorded fields relative to the optic disk
        -> ExpInfo
        -> Field
        ---
        relx        :float      # x position relative to the optic disk
        rely        :float      # y position relative to the optic disk
        relz        :float      # z position relative to the optic disk
        """

    def _make_tuples(self, key):
        date = key["date"]
        date = date.strftime('%Y-%m-%d')
        experimenter = key["experimenter"]
        exp_num = key["exp_num"]
        # if the fields from this experiment (identified by date and exp_num) do not yet have the optic disk information:
        if np.size((RelativeFieldLocation() & 'date="{0}" and exp_num={1} and experimenter="{2}"'.format(date, exp_num,
                                                                                                         experimenter)).fetch(
            "relx")) == 0:

            odx = 0
            ody = 0
            odz = 0

            # get a table of all fields that belong to the same experiment
            table = pd.DataFrame((
                                         Field().RoiMask() * Field() & 'date="{0}" and exp_num={1} and experimenter="{2}"'.format(
                                     date, exp_num, experimenter)).fetch())

            # use the optic disk locations read in from the .ini file in Experiment().ExpInfo()
            if (ExpInfo() & key).fetch1("od_valid_flag") == 1:
                odx = (ExpInfo() & key).fetch1("ody")  # swap x & y axis for od coordinates
                ody = (ExpInfo() & key).fetch1("odx")  # swap x & y axis for od coordinates
                odz = (ExpInfo() & key).fetch1("odz")

            interm_absx = copy.deepcopy(table["absx"])

            eye = (ExpInfo() & key).fetch1('eye') #here use the eye information rather than the experiment number
            if eye == 'left':
                table["absx"] = -1 * (table["absy"] - odx)  # swap x & y axis for field coordinates and multiply with -1 for LR
            else:
                table["absx"] = table["absy"] - odx  # swap x & y axis for field coordinates

            if experimenter == 'Franke': # Franke's dorsal retina close to the experimenter
                table["absy"] = interm_absx - ody  #swap x & y axis for field coordinates & make dorsal retina with positive sign
            else: # Korympidou's dorsal retina close to the back of the chamber
                table["absy"] = -1 * (
                    interm_absx - ody)  # swap x & y axis for field coordinates & make dorsal retina with positive sign

            table["absz"] = table["absz"] - odz

            for i in range(len(table["absx"])):
                key["relx"] = table["absx"][i]
                key["rely"] = table["absy"][i]
                key["relz"] = table["absz"][i]
                key["field_id"] = table["field_id"][i]
                try:
                    self.insert1(key)
                except:
                    print(key)


@schema
class QualityLocation(dj.Computed):
    definition = """
    # Information about recording of optic disk
    -> Experiment
    ---
    quality_location  :int                # 0 if optic disk recording was incorrect -> Don’t use field location info
    """

    def _make_tuples(self, key):

        # Optic nerve location was recorded incorrectly for these experiments
        # Based on manual notes by Marili
        # Occured for drug experiments when retina was cut
        incorrect_fields = [{'experimenter': 'Korympidou',
                             'date': datetime.date(2022,4,13),
                             'exp_num': 2},
                            {'experimenter': 'Korympidou',
                             'date': datetime.date(2022,5,9),
                             'exp_num': 1},
                            {'experimenter': 'Korympidou',
                             'date': datetime.date(2022,5,11),
                             'exp_num': 4},
                            {'experimenter': 'Korympidou',
                             'date': datetime.date(2022,5,13),
                             'exp_num': 1},
                            {'experimenter': 'Korympidou',
                             'date': datetime.date(2022,5,19),
                             'exp_num': 2},
                            {'experimenter': 'Korympidou',
                             'date': datetime.date(2022,5,19),
                             'exp_num': 3}]

        current_field = {'experimenter': key['experimenter'],
                        'date': key['date'],
                        'exp_num': key['exp_num']}
        if np.isin(current_field, incorrect_fields):
            quality = 0 # Bad experiment
        else:
            quality = 1 # Good experiment

        # add to the table
        self.insert1(
            dict(key,
                 quality_location=quality,
                 ))


@schema
class NormalizedChirp(dj.Computed):
    definition = """
        # Preprocessing of chirp traces for clustering
        ->SummarySnippets
        roi               :smallint       #roi number in the field
        ---
        time_bin          :blob           #list of time points corresponding to chirp
        normalized_chirp  :blob           #normalized and baselined response to the chirp
        """

    @property
    def key_source(self):
        rel = super().key_source
        return rel * (Stimulus & 'stim_id in (5,6)').proj()

    def _make_tuples(self, key):
        summarydata = (SummarySnippets() & key).fetch1()

        columns = summarydata['condition_average_strs'].split()
        data_df = pd.DataFrame(columns=columns)
        data_stack = summarydata['condition_average_frame']
        data_df = pd.concat([data_df, pd.DataFrame(data_stack, columns=columns)])
        data_df = data_df.astype({'roi_ids': 'int32'})
        data_df = data_df.rename(
            columns={'roi_ids': 'rois', 'time_bin': 'time_bin_chirp', 'preprocess_snippets': 'chirp'})
        length_baseline = (ResamplingFrequency() & key).fetch1('resample_f') #update this to use the sampling frequency. Now we get 1 second for baseline
        grouped_df = data_df.groupby(['rois', 'rois'])

        for index, group in grouped_df:  # code from Sarah's notebooks for normalizing the chirp
            current_roi = index[0]
            current_bin = group['time_bin_chirp'].values
            current_chirp = group['chirp'].values

            chirp_baseline = current_chirp - np.mean(current_chirp[0:length_baseline])
            normalization = np.amax((np.amax(chirp_baseline), np.abs(np.amin(chirp_baseline))))
            chirp_norm = chirp_baseline / normalization
            key['roi'] = current_roi

            self.insert1(dict(key,
                              time_bin=current_bin,
                              normalized_chirp=chirp_norm
                              ))


@schema
class BC_ColorKernels(dj.Computed):
    definition = """
    # UV kernels calculated from UV flicker
    ->Field
    ->PreprocessTraces
    ->BC_NoiseStimulus
    ->BGFlickerParams
    ---
    interpolated_traces_uv                  :longblob         # traces obtained from linear interpolation (roi x time)
    cropped_traces_uv                       :longblob         # traces cropped to the stimuluated periods
    cropped_stimulus_uv                     :longblob         # stimulus cropped to the stimulated period
    st_kernels_uv                           :longblob         # space-time kernels (space, line, roi) for uv
    offset_before_uv                        :int              # line offset used in kernel and convolution calculations
    offset_after_uv                         :int              # line offset used in kernel and convolution calculations
    kernel_length_line_uv                   :int              # kernel length in lines
    """

    class BC_ColorKernels_Green(dj.Part):
        definition = """
        # Green kernels calculated from green flicker
        ->BC_ColorKernels
        ---
        interpolated_traces_green               :longblob         # traces obtained from linear interpolation (roi x time)
        cropped_traces_green                    :longblob         # traces cropped to the stimuluated periods
        cropped_stimulus_green                  :longblob         # stimulus cropped to the stimulated period
        st_kernels_green                        :longblob         # space-time kernels (space, line, roi) for green
        offset_before_green                     :int              # line offset used in kernel and convolution calculations
        offset_after_green                      :int              # line offset used in kernel and convolution calculations
        kernel_length_line_green                :int              # kernel length in lines
        """

    @property
    def key_source(self):
        rel = super().key_source
        return rel * Stimulus() & ['stimulusname = "blue_csflicker"']

    def _make_tuples(self, key):
        stimulus_name = (Stimulus() & key).fetch1('stimulusname')
        # get kernels for the uv stimulus
        interpolated_traces_uv, stimulus_uv  = self.interpolate_traces(key)
        st_kernels_uv, cropped_traces_uv, cropped_stimulus_uv, offset_before_uv, offset_after_uv, kernel_length_line_uv = self.matrix_multiply(key, interpolated_traces_uv, stimulus_uv)

        self.insert1(dict(key,
                          interpolated_traces_uv=interpolated_traces_uv,
                          cropped_traces_uv=cropped_traces_uv,
                          cropped_stimulus_uv=cropped_stimulus_uv,
                          st_kernels_uv=st_kernels_uv,
                          offset_before_uv=offset_before_uv,
                          offset_after_uv=offset_after_uv,
                          kernel_length_line_uv=kernel_length_line_uv
                          ))

        primary_key = deepcopy(key)
        # set up key to get the green field
        keys_to_extract = ['experimenter', 'date', 'exp_num', 'field_id', 'noise_id']
        green_key = {key1: primary_key[key1] for key1 in keys_to_extract}
        stim_ids = ['stim_id = 17']
        green_stim_id = (PreprocessTraces() & green_key & stim_ids).fetch1('stim_id')
        green_key['stim_id'] = green_stim_id
        # get kernels for the green stimulus
        interpolated_traces_green, stimulus_green  = self.interpolate_traces(green_key)
        st_kernels_green, cropped_traces_green, cropped_stimulus_green, offset_before_green, offset_after_green, kernel_length_line_green = self.matrix_multiply(green_key, interpolated_traces_green, stimulus_green)

        BC_ColorKernels.BC_ColorKernels_Green().insert1(dict(primary_key,
                                                          interpolated_traces_green=interpolated_traces_green,
                                                          cropped_traces_green=cropped_traces_green,
                                                          cropped_stimulus_green=cropped_stimulus_green,
                                                          st_kernels_green=st_kernels_green,
                                                          offset_before_green=offset_before_green,
                                                          offset_after_green=offset_after_green,
                                                          kernel_length_line_green=kernel_length_line_green))

    def interpolate_traces(self, key):
        # get trace data
        line_duration = (Presentation() & key).fetch1('scan_line_duration')
        traces = (PreprocessTraces() & key).fetch1('preprocess_traces')
        traces_times_all = (TraceTriggerTimes() & key).fetch1('trace_times')
        stimulus = (BC_NoiseStimulus() & key).fetch1('noise_array_line')    #already at line precision

        # do linear interpolation of the traces
        # set up time array
        tstart = 0
        nlines = stimulus.shape[1]
        tend = line_duration * nlines
        timeALL = np.linspace(tstart, tend, nlines)
        # make arrays for linear interpolation
        yold = traces.T
        xold = traces_times_all.T
        nrowsold = yold.shape[0]  # number of rows in the x_t and y traces
        timenew = np.tile(timeALL, (nrowsold, 1))
        # the interpolation function
        arraytuples = [*zip(timenew, xold, yold)]
        pool = Pool(os.cpu_count())  # use multiple nodes to iterate over the data
        result = (pool.starmap(np.interp, iterable=arraytuples))  # result contains the interpolated y values
        # result is list of arrays with each item representing a roi
        ynew = np.array(result)
        return ynew, stimulus


    def matrix_multiply(self, key, traces, stimulus):#traces, stimulus, offset_before, offset_after,kernel_length_line):  # get the kernels

        #get the lines when stimulus started and stopped
        triggertimes = (TraceTriggerTimes() & key).fetch1('trigger_times_corrected')
        line_duration = (Presentation() & key).fetch1('scan_line_duration')
        trigger_starts_lines = np.floor(triggertimes[0] / line_duration)
        trigger_ends_lines = np.ceil(triggertimes[-1] / line_duration)

        # crop traces and stimuli to only include the stimulated periods
        traces_for_kernel = traces[:,np.int(trigger_starts_lines):np.int(trigger_ends_lines+1)]
        cropped_stimulus = stimulus[:, np.int(trigger_starts_lines):np.int(trigger_ends_lines+1)]

        kernel_params = (BGFlickerParams & key).fetch1()
        kernel_length_s = kernel_params['filter_length']
        kernel_length_line = np.int(np.floor(kernel_length_s / line_duration))
        offset_after = np.int(np.floor(kernel_length_line * .25))  # lines to include into the future (using 1/4 of kernel length)
        offset_before = kernel_length_line - offset_after

        full_traces = traces_for_kernel[:, offset_before:-offset_after]
        full_matrix = np.zeros((kernel_length_line, full_traces.shape[1], cropped_stimulus.shape[0]))
        trace_length = full_traces.shape[1]
        for i in range(kernel_length_line):
            full_matrix[i, :, :] = cropped_stimulus[:, i:trace_length + i].T
        # Z scoring traces and stimulus matrix
        for i in range(2):
            full_matrix[:, :, i] = (full_matrix[:, :, i].T - np.mean(full_matrix[:, :, i], axis=1)).T
            full_matrix[:, :, i] = (full_matrix[:, :, i].T / np.std(full_matrix[:, :, i], axis=1)).T
        full_traces = (full_traces.T - np.mean(full_traces, axis=1)).T
        full_traces = (full_traces.T / np.std(full_traces, axis=1)).T
        kernels_all_rois = np.matmul(full_traces, full_matrix)
        kernels_all_rois /=full_traces.shape[1]
        return kernels_all_rois, traces_for_kernel, cropped_stimulus, offset_before, offset_after, kernel_length_line


@schema
class BC_MinMaxQuality(dj.Computed):
    definition = """
    # Quality filtering of kernels based on the max minus min
    ->MinMaxQualityParams
    ->BC_ColorKernels
    ---
    quality_values            :blob               # maximum amplitude in each kernel
    """
    def _make_tuples(self, key):
        # get the kernels
        kernels_uv_all_rois = (BC_ColorKernels & key).fetch1('st_kernels_uv')
        kernels_green_all_rois = (BC_ColorKernels.BC_ColorKernels_Green & key).fetch1('st_kernels_green')

        # get parameter for kernel window
        start_kernel = (MinMaxQualityParams & key).fetch1('start_kernel')
        stop_kernel = int(kernels_green_all_rois.shape[0]*3/4) #stop at zero time

        # Calculate max and min values during response period
        min_uv = np.amin(kernels_uv_all_rois[start_kernel:stop_kernel, :, :], axis=0)
        max_uv = np.amax(kernels_uv_all_rois[start_kernel:stop_kernel, :, :], axis=0)
        min_green = np.amin(kernels_green_all_rois[start_kernel:stop_kernel, :, :], axis=0)
        max_green = np.amax(kernels_green_all_rois[start_kernel:stop_kernel, :, :], axis=0)

        # Calculate kernel amplitude
        amplitude_uv = np.abs(max_uv - min_uv)
        amplitude_green = np.abs(max_green - min_green)

        # Take largest amplitude across conditions; Do *not* include UV surround as it has the artefact
        max_amplitude = np.amax(np.concatenate((amplitude_uv, amplitude_green), axis=1), axis=1)

        self.insert1(dict(key,
                          quality_values=max_amplitude,
                          ))


@schema
class BC_IplDepth(dj.Computed):
    definition = """
        # Gives the IPL depth of the field's ROIs relative to the GCL (=0) and INL (=1)
        ->Field
        ---
        depth         :blob           #Depth in the IPL ordered by ROI
        """

    def _make_tuples(self, key):

        data_directory = (UserInfo & key).fetch("data_dir")[0] + 'RoiDepths/'
        file_name = key['date'].strftime('%Y%m%d') + '_' + str(key['exp_num']) + '_' + str(key['field_id']) + '.txt'

        with open(data_directory+file_name) as file:
            lines = file.readlines()
            lines = [line.split() for line in lines]

        depths = np.zeros(len(lines)-1)
        for index, current_line in enumerate(lines[1:]):
            depths[index] = np.asarray([float(i) for i in current_line])

        # Linear transformation y = k*x + d;
        # d = 1.1/3; k = 1/3;
        # x position in BC data where 0 1 are ChAT bands
        # y position in AC data where 0 1 are GCL and INL
        # From https://www.nature.com/articles/s41598-020-60214-z ** Figure 3 **
        depths_new_coordinate = (depths + 1.1)/3

        self.insert1(dict(key,
                          depth=depths_new_coordinate))
