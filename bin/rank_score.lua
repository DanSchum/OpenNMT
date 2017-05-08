require('onmt.init')

local cmd = onmt.utils.ExtendedCmdLine.new('rank_score.lua')

local options = {
  {'-src', '', [[Source sequence to decode (one line per sequence)]]},
  {'-tgt', '', [[Nbest list for the target side]], {valid=onmt.utils.ExtendedCmdLine.nonEmpty}},
  {'-output', 'pred.txt', [[Path to output the new n_best list]]}
}

cmd:setCmdLineOptions(options, 'Data')

cmd:text('')
cmd:text('**Other options**')
cmd:text('')

onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)

local function reportScore(name, scoreTotal, wordsTotal)
  _G.logger:info(name .. " AVG SCORE: %.2f, " .. name .. " PPL: %.2f",
                 scoreTotal / wordsTotal,
                 math.exp(-scoreTotal/wordsTotal))
end

local function main()
  local opt = cmd:parse(arg)

  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)

  local srcReader = onmt.utils.FileReader.new(opt.src)
  --~ local srcBatch = {}
  
  local hypReader = onmt.utils.FileReader.new(opt.tgt)
  

  local outFile = io.open(opt.output, 'w')
  local sentId = 0
  local batchId = 1

  local predScoreTotal = 0
  local predWordsTotal = 0
	
	
	local hypID = -1
	local srcSent
	local tgtNBest = {}
	local scores = {}
	
	while true do
		
		local currentTgtTokens = hypReader:next()
		
		-- end of file
    if currentTgtTokens == nil then
			-- proceed this batch and print out result
			--~ local results = rescorer:rescore(srcSent, tgtNBest)
			
			local listSize = #tgtNBest
			local score = torch.Tensor(listSize):zero()
			
			-- Accumulate the score: using sum
			for n = 1, listSize do
				for m = 1, #tgtScores[1] do
					
					-- Nematus score is negative
					if tgtScores[n][m] > 0 then
						tgtScores[n][m] = - tgtScores[n][m]
					end
					score[n] = score[n] + tgtScores[n][m]
				end 
			end
			
			-- sort (descending)
			local sorted_score, sorted_id = torch.sort(score, 1, true) 
			
			local bestScore = sorted_score[1]
			local bestSentence = tgtNBest[sorted_id[1]]
			
			local sentWithScore = string.format("%s : %.2f", bestSentence, bestScore)
			_G.logger:info("BEST HYP: ")
			_G.logger:info(sentWithScore)
			outFile:write(bestSentence .. "\n")
			break
		end
		
		local length = #currentTgtTokens
		
		if length > 0 then
			-- find the position of the last "|||"
			local lastMarker
			for j = length, 1, -1 do
				if currentTgtTokens[j] == '|||' then
					lastMarker = j
					break
				end
			end
			
			-- build sentence
			local sentTokens = {}
			for j = 3, lastMarker -1 do
				table.insert(sentTokens, currentTgtTokens[j])
			end
			
			
			local scores = {}
			for j = lastMarker + 1, length do
				table.insert(scores, tonumber(currentTgtTokens[j]))
			end
			
			local tgtHyp = table.concat(sentTokens, ' ')
			
			local currentHypId = tonumber(currentTgtTokens[1])
			
			-- end of last sentence -> proceed this batch
			if currentHypId > hypID then
				
				if hypID > -1 then
					-- proceed this batch and print out result
					--~ local results = rescorer:rescore(srcSent, tgtNBest)
					
					local listSize = #tgtNBest
					local score = torch.Tensor(listSize):zero()
					
					-- Accumulate the score: using sum
					for n = 1, listSize do
						for m = 1, #tgtScores[1] do
							
							-- Nematus score is negative
							if tgtScores[n][m] > 0 then
								tgtScores[n][m] = - tgtScores[n][m] * 0.01
							end
							
							if m > 1 then
								tgtScores[n][m] = tgtScores[n][m] * 0.5
							end
							
							score[n] = score[n] + tgtScores[n][m]
						end 
					end
					
					-- sort (descending)
					local sorted_score, sorted_id = torch.sort(score, 1, true) 
					
					local bestScore = sorted_score[1]
					local bestSentence = tgtNBest[sorted_id[1]]
					
					local sentWithScore = string.format("%s : %.2f", bestSentence, bestScore)
					_G.logger:info("BEST HYP: ")
					_G.logger:info(sentWithScore)
					outFile:write(bestSentence .. "\n")
				end
				
				-- build next batch
				tgtNBest = {tgtHyp}
				tgtTokens = {currentTgtTokens}				
				if hypID == -1 then 
					hypID = currentHypId
				else
					hypID = hypID + 1
				end
				tgtScores = {scores}
			else
				table.insert(tgtNBest, tgtHyp)
				table.insert(tgtScores, scores)
				table.insert(tgtTokens, currentTgtTokens)
			end
		
		end
		--~ while true do
			--~ local nbestList = {}
			--~ local srcTokens = srcReader:next()
			
			--~ -- end of file
			--~ if srcTokens == nil then
				--~ break
			--~ end
			
			--~ hypID = 
			
			--~ for n = 1, nbestSize do
			--~ while true
				--~ local currentTgtTokens = hypReader:next()
				--~ currentHypId = currentTgtTokens[1]
				--~ local currentHypScore = currentTgtTokens[#currentTgtTokens]
				
				--~ local sentTokens = {}
				
				--~ for i = 3, #currentTgtTokens - 2 do
					--~ table.insert(sentTokens, currentTgtTokens[i])
				--~ end
				
				--~ local sentence = rescorer:buildInput(sentTokens)
				--~ table.insert(nbestList, sentence)
			--~ end
			
			
			--~ local srcSent = rescorer:buildInput(srcTokens)
			
			--~ local results = rescorer:rescore(srcSent, nbestList)
			
			
			--~ for n = 1, nbestSize do
				--~ local sentId = currentHypId
				--~ local sentence = results[n].sent
				--~ local score = results[n].score
				
				--~ local sentWithScore = string.format("%i ||| %s ||| %.2f", sentId, sentence, score)
				
				--~ outFile:write(sentWithScore .. '\n')
				--~ _G.logger:info(sentWithScore)
			--~ end
			
			--~ outFile:write('\n')
			--~ _G.logger:info('')
			
			
			--~ -- increase the current sentence by 1
			--~ currentHypId = currentHypId + 1
		

		end
  
  outFile:close()
  _G.logger:shutDown()
end

main()
