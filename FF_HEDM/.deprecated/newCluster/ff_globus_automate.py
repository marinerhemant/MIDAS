from globs_automate_client import (create_flows_client, graphviz_format, state_colors_for_log, create_action_client, create_flows_client)
from globus_automate_client.token_management import CLIENT_ID
import time, json, sys, os, copy
from funcx.sdk.client import FuncXClient
import datetime
from utils import *
from remote_funcs import *
import argparse

fxc = FuncXClient()
func_prepare = fxc.register_function(remote_prepare)
func_peaksearch = fxc.register_function(remote_peaksearch)
func_transforms = fxc.register_function(remote_transforms)
func_indexrefine = fxc.register_function(remote_indexrefine)
func_grains = fxc.register_function(remote_find_grains)

# Need to read the file and get whatever we need:
# paramFileName startLayerNr endLayerNr nFrames numProcs numBlocks timePath StartFileNrFirstLayer NrFilesPerSweep FileStem SeedFolder StartNr EndNr '''blockNr'''
# paramFileName must not have full path
# paramFile must not have SeedFolder or RawFolder names
# paramFile must have a parameter named RawDir: location of the source data.
# RawDir should contain the rawFiles (only the ones we need for this analysis: data + dark + parameterFile).

parser = argparse.ArgumentParser(description='''MIDAS_FF, contact hsharma@anl.gov Parameter file must be in the same folder as the desired output folder(SeedFolder)''', formatter_class=RawTextHelpFormatter)
parser.add_argument('-nCPUs',    type=int, required=True, help='Number of CPU cores per node to use')
parser.add_argument('-numNodes',    type=int, required=True, help='Number of nodes to use')
parser.add_argument('-startLayerNr',type=int,required=True,help='Start Layer Number')
parser.add_argument('-endLayerNr',type=int,required=True,help='End Layer Number')
parser.add_argument('-paramFile', type=str, required=True, help='ParameterFileName')
args, unparsed = parser.parse_known_args()
paramFN = args.paramFile
startLayerNr = int(args.startLayerNr)
endLayerNr = int(args.endLayerNr)
numProcs = int(args.nCPUs)
numBlocks = int(args.numNodes)
thisT = datetime.datetime.now()
tod = datetime.date.today()
timePath = str(tod.year) + '_' + str(tod.month).zfill(2) + '_' + str(tod.day).zfill(2) + '_' + str(thisT.hour).zfill(2) + '_' + str(thisT.minute).zfill(2) + '_' + str(thisT.second).zfill(2)
paramContents = open(paramFN).readlines()
for line in paramContents:
	if line.startswith('StartFileNrFirstLayer'):
		startNrFirstLayer = int(line.split()[1])
	if line.startswith('NrFilesPerSweep'):
		nrFilesPerSweep = int(line.split()[1])
	if line.startswith('FileStem'):
		fileStem = line.split()[1]
	if line.startswith('StartNr'):
		startNr = int(line.split()[1])
	if line.startswith('EndNr'):
		endNr = int(line.split()[1])
	if line.startswith('RawDir'):
		endNr = int(line.split()[1])
nFrames = endNr - startNr + 1

# Need to set up paths:
'''
sourcePath: folder: we setup a folder for each analysis, to copy the whole folder
executePath: folder
executeResultPath: we generate a tar file of the whole folder (recon_time_path.tar.gz)
resultPath: folder + tar (recon_time_path.tar.gz)
'''
sourcePath = rawDir
executePath = '/lus/theta-fs0/projects/APSPolarisI2E/HEDM/'
executeResultPath = executePath+'recon_'+timePath+'.tar.gz'
resultPath = sourcePath+'recon_'+timePath+'.tar.gz'
seedFolder = executePath

# EP Names (not needed to change very frequently)
sourceEP = aps_data_ep # options: aps_data_ep, clutch_ep or give the ID of where your data lives (see utils.py for examples)
destEP = aps_data_ep # options: aps_data_ep, clutch_ep or give the ID of where your data lives (see utils.py for examples)
remoteDataEP = theta_glob_ep # options: this can also be some other location where you might want to store the data during remote execute. NB: different from the next parameter
remoteExecuteEP = fxc_ep # options: this can be any location where you have started a funcX endpoint and have permissions to run analysis.

from setup_payloads import *
print (flow_input)
from ff_flow import *
print(flow_definition)

flows_client = create_flows_client(CLIENT_ID)
flow = flows_client.deploy_flow(flow_definition,title="MIDAS FF-HEDM workflow")
flow_id = flow['id']
print(flow)
flow_scope = flow['globus_auth_scope']
print(f'Flow created with ID: {flow_id}\nScope: {flow_scope}')

flow_action = flows_client.run_flow(flow_id,flow_scope,flow_input)
print(flow_action)
flow_action_id = flow_action['action_id']
flow_status = flow_action['status']
print(f'Flow action started with id: {flow_action_id}')
while flow_status == 'ACTIVE':
	time.sleep(10)
	flow_action = flows_client.flow_action_status(flow_id, flow_scope, flow_action_id)
	flow_status = flow_action['status']
	print(f'Flow status: {flow_status}')

print(flow_action)
