class SetupPayloads():
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
	flow_input = {
		"input": {
			"source_endpoint":sourceEP,
			"source_path":sourcePath,
			"execute_endpoint":remoteDataEP,
			"execute_path":executePath,
			"execute_result_path":executeResultPath,
			"result_endpoint":destEP,
			"result_path":resultPath,
			"paramFileName":pfName,
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
			"EndNr":endNr,}
		}
	flow_input['input'].update('tasks_multiple':[{
			'startLayerNr':'$.input.startLayerNr',
			'endLayerNr':'$.input.endLayerNr',
			'numProcs':'$.input.numProcs',
			'numBlocks':'$.input.startLayerNr',
			'blockNr':f'{idx}',
			'timePath':'$.input.timePath',
			'FileStem':'$.input.FileStem',
			'SeedFolder':'$.input.SeedFolder',
			'paramFileName':'$.input.paramFileName',
			}] for idx in range(numBlocks))
	flow_input['input'].update('indexrefine_tasks':[{
			'endpoint.$':'$.input.funcx_endpoint_compute',
			'function.$':'$.input.remote_indexrefine_funcx_id',
			'payload.$':f'$.input.tasks_multiple[{idx}]'
		} for idx in range(numBlocks)])
	flow_input['input'].update('peaksearch_tasks':[{
			'endpoint.$':'$.input.funcx_endpoint_compute',
			'function.$':'$.input.remote_peaksearch_funcx_id',
			'payload.$':f'$.input.tasks_multiple[{idx}]'
		} for idx in range(numBlocks)])
