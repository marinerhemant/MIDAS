DataIn = {
	"Comment":"Inject data",
	"Type":"Action",
	"ActionUrl":"https://actions.automate.globus.org/transfer/transfer",
	"ActionScope":"https://auth.globus.org/scopes/actions.globus.org/transfer/transfer",
	"Parameters":{
		"source_endpoint_id.$":"$.input.source_endpoint",
		"destination_endpoint_id.$": "$.input.execute_endpoint",
		"transfer_items": [
			{
				"source_path.$": "$.input.source_path",
				"destination_path.$": "$.input.execute_path",
				"recursive": True
			}
		]
	},
	"ResultPath": "$.InjectDataResult",
	"WaitTime": 1200,
	"Next": "remote_prepare"
}

DataOut = {
	"Comment":"Extract data",
	"Type":"Action",
	"ActionUrl":"https://actions.automate.globus.org/transfer/transfer",
	"ActionScope":"https://auth.globus.org/scopes/actions.globus.org/transfer/transfer",
	"Parameters":{
		"source_endpoint_id.$":"$.input.execute_endpoint",
		"destination_endpoint_id.$": "$.input.result_endpoint",
		"transfer_items": [
			{
				"source_path.$": "$.input.execute_result_path",
				"destination_path.$": "$.input.result_path",
				"recursive": True
			}
		]
	},
	"ResultPath": "$.InjectDataResult",
	"WaitTime": 1200,
	"End": True
}

FxPrepare = {
	"Comment":"Prepare for remote",
	"Type":"Action",
	"ActionUrl":"https://api.funcx.org/automate",
	"ActionScope": "https://auth.globus.org/scopes/facd7ccc-c5f4-42aa-916b-a0e270e2c2a9/all",
	"InputPath":"$.prepareinput",
	"ResultPath":"$.PrepareResult"
	"WaitTime":1200,
	"Next":"remote_peaksearch"
}

FxPeakSearch = {
	"Comment":"Remote peaksearch",
	"Type":"Action",
	"ActionUrl":"https://api.funcx.org/automate",
	"ActionScope": "https://auth.globus.org/scopes/facd7ccc-c5f4-42aa-916b-a0e270e2c2a9/all",
	"InputPath":"$.peaksearchinput",
	"ResultPath":"$.PeakSearchResult"
	"WaitTime":1200,
	"Next":"remote_transforms"
}

FxTransforms = {
	"Comment":"Remote transforms",
	"Type":"Action",
	"ActionUrl":"https://api.funcx.org/automate",
	"ActionScope": "https://auth.globus.org/scopes/facd7ccc-c5f4-42aa-916b-a0e270e2c2a9/all",
	"InputPath":"$.transformsinput",
	"ResultPath":"$.TransformsResult"
	"WaitTime":1200,
	"Next":"remote_indexrefine"
}

FxIndexRefine = {
	"Comment":"Remote Index Refine",
	"Type":"Action",
	"ActionUrl":"https://api.funcx.org/automate",
	"ActionScope": "https://auth.globus.org/scopes/facd7ccc-c5f4-42aa-916b-a0e270e2c2a9/all",
	"InputPath":"$.indexrefineinput",
	"ResultPath":"$.IndexRefineResult"
	"WaitTime":1200,
	"Next":"remote_grains"
}

FxGrains = {
	"Comment":"Remote Grains compute",
	"Type":"Action",
	"ActionUrl":"https://api.funcx.org/automate",
	"ActionScope": "https://auth.globus.org/scopes/facd7ccc-c5f4-42aa-916b-a0e270e2c2a9/all",
	"InputPath":"$.grainsinput",
	"ResultPath":"$.GrainsResult"
	"WaitTime":1200,
	"Next":"data_out"
}

flow_definition = {
	"Comment":"MIDAS-FF flow",
	"StartAt":"data_in"
	}
flow_definition["States"] = {}
flow_definition["States"]["data_in"] = copy.deepcopy(DataIn)
flow_definition["States"]["data_out"] = copy.deepcopy(DataOut)
flow_definition["States"]["remote_prepare"] = copy.deepcopy(FxPrepare)
flow_definition["States"]["remote_peaksearch"] = copy.deepcopy(FxPeakSearch)
flow_definition["States"]["remote_transforms"] = copy.deepcopy(FxTransforms)
flow_definition["States"]["remote_indexrefine"] = copy.deepcopy(FxIndexRefine)
flow_definition["States"]["remote_grains"] = copy.deepcopy(FxGrains)
flow_definition = json.dumps(flow_definition)
