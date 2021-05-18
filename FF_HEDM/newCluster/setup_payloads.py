common_payload = {
	"paramFileName.$":"$.input.pfname",
	"startLayerNr.$":"$.input.startLayerNr",
	"endLayerNr.$":"$.input.endLayerNr",
	"nFrames.$":"$.input.nFrames",
	"numProcs.$":"$.input.numProcs",
	"numBlocks.$":"$.input.numBlocks",
	"timePath.$":"$.input.timePath",
	"StartNrFirstLayer.$":"$.input.StartFileNrFirstLayer",
	"NrFilesPerSweep.$":"$.input.NrFilesPerSweep",
	"FileStem.$":"$.input.FileStem",
	"SeedFolder.$":"$.input.SeedFolder",
	"StartNr.$":"$.input.StartNr",
	"EndNr.$":"$.input.EndNr"}

# Payload for prepare
prepareinput = {}
prepareinput["tasks"] = [{"endpoint.$": "$.input.fx_ep",
					"func.$": "$.input.fx_prepare"}]
prepareinput["tasks"]["payload"] = copy.deepcopy(common_payload)

# Setup payload for peaksearch
tasks = {}
alltasks = []
for nodeNr in range(nNodes):
	thistask = {"endpoint.$": "$.input.fx_ep",
					"func.$": "$.input.fx_peaksearch"}
	thispayload = {}
	thispayload = copy.deepcopy(common_payload)
	thispayload["blockNr"] = nodeNr
	thistask["payload"] = thispayload
	alltasks.append(thistask)
tasks["tasks"] = alltasks
peaksearchinput = copy.deepcopy(tasks)

# Payload for transforms
transformsinput = {}
transformsinput["tasks"] = [{"endpoint.$": "$.input.fx_ep",
					"func.$": "$.input.fx_transforms"}]
transformsinput["tasks"]["payload"] = copy.deepcopy(common_payload)

# setup payload for IndexRefine
tasks = {}
alltasks = []
for nodeNr in range(nNodes):
	thistask = {"endpoint.$": "$.input.fx_ep",
					"func.$": "$.input.fx_indexrefine"}
	thispayload = {}
	thispayload = copy.deepcopy(common_payload)
	thispayload["blockNr"] = nodeNr
	thistask["payload"] = thispayload
	alltasks.append(thistask)
tasks["tasks"] = alltasks
indexrefineinput = copy.deepcopy(tasks)

# Payload for grains
grainsinput = {}
grainsinput["tasks"] = [{"endpoint.$": "$.input.fx_ep",
					"func.$": "$.input.fx_grains"}]
grainsinput["tasks"]["payload"] = copy.deepcopy(common_payload)


flowinput = {}
flowinput["input"] = {
		"source_endpoint":sourceEP,
		"source_path":sourcePath,
		"execute_endpoint":remoteDataEP,
		"execute_path":executePath,
		"execute_result_path":executeResultPath,
		"result_endpoint":destEP,
		"result_path":resultPath,
		"fx_ep":remoteExecuteEP,
		"fx_prepare":func_prepare,
		"fx_peaksearch":func_peaksearch,
		"fx_transforms":func_transforms,
		"fx_indexrefine":func_indexrefine,
		"fx_grains":func_grains,
		"pfname":pfName,
		"startLayerNr":startLayerNr,
		"endLayerNr":endLayerNr,
		"nFrames":nFrames,
		"numProcs":numProcs,
		"numBlocks":numBlocks,
		"timePath":timePath,
		"StartFileNrFirstLayer":startNrFirstLayer,
		"NrFilesPerSweep":nrFilesPerSweep,
		"FileStem":fileStem,
		"SeedFolder":seedFolder,
		"StartNr":startNr,
		"EndNr":endNr,
		}
flowinput["input"]["prepareinput"] = copy.deepcopy(prepareinput)
flowinput["input"]["peaksearchinput"] = copy.deepcopy(peaksearchinput)
flowinput["input"]["transformsinput"] = copy.deepcopy(transformsinput)
flowinput["input"]["indexrefineinput"] = copy.deepcopy(indexrefineinput)
flowinput["input"]["grainsinput"] = copy.deepcopy(grainsinput)

flow_input = json.dumps(flowinput)
