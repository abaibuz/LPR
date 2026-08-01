//
//  Untitled.swift
//  YOLO
//
//  Created by Oleksandr Baibuz on 20.12.2025.
//  Copyright © 2025 Ultralytics. All rights reserved.
//

import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision
import TesseractOCR
import AudioToolbox

extension ViewController {
    
    //
    func suppressDuplicateBoxes(boxes: [CGRect], iouThreshold: CGFloat = 0.7) -> [CGRect] {
        
        var result: [CGRect] = []
        
        for box in boxes {
            var isDuplicate = false
            
            for kept in result {
                if iou(box, kept) > iouThreshold {
                    isDuplicate = true
                    break
                }
            }
            
            if !isDuplicate {
                result.append(box)
            }
        }
        
        return result
    }
    
    //IoU для сопоставления рамок
    func iou(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let inter = a.intersection(b)
        guard !inter.isNull else { return 0 }
        let interArea = inter.width * inter.height
        let unionArea = a.width*a.height + b.width*b.height - interArea
        return interArea / unionArea
    }
    
    //----------
    func centerDistance(_ a: CGRect, _ b: CGRect) -> CGFloat {
        let dx = a.midX - b.midX
        let dy = a.midY - b.midY
        return sqrt(dx*dx + dy*dy)
    }
    
    
    //---------
    func levenshtein(_ aStr: String, _ bStr: String) -> Int {
        let a = Array(aStr)
        let b = Array(bStr)
        
        var dist = Array(repeating: Array(repeating: 0, count: b.count + 1), count: a.count + 1)
        
        for i in 0...a.count { dist[i][0] = i }
        for j in 0...b.count { dist[0][j] = j }
        
        for i in 1...a.count {
            for j in 1...b.count {
                if a[i-1] == b[j-1] {
                    dist[i][j] = dist[i-1][j-1]
                } else {
                    dist[i][j] = min(
                        dist[i-1][j] + 1,
                        dist[i][j-1] + 1,
                        dist[i-1][j-1] + 1
                    )
                }
            }
        }
        
        return dist[a.count][b.count]
    }
    
    //----------
    func similarity(_ a: String, _ b: String) -> CGFloat {
        if a.isEmpty || b.isEmpty { return 0 }
        
        let distance = levenshtein(a, b)
        let maxLen = max(a.count, b.count)
        
        return 1.0 - CGFloat(distance) / CGFloat(maxLen)
    }
    
    
    //------Track Matching
    func matchTrackMy(for box: CGRect, boxIndex: Int, plates: [String]) -> (Int,String) {
        var threshold: Double = 0.5
        if Thread.isMainThread {
            threshold = Double(round(100 * iOuTreshold)) / 100
        } else {
            DispatchQueue.main.sync {
                threshold = Double(round(100 * iOuTreshold)) / 100
            }
        }
        let adaptiveDist = max(box.width, box.height) * 3
        let adaptiveDistDebug = CGFloat(self.centerDistanceThreshold)
        
        let centerDistDebug = max(adaptiveDist,adaptiveDistDebug)
        
        var ocrScore: CGFloat = 0
        let ocrScoreThreshold = CGFloat(ocrThreshold)
        
        var normalizedPlate: String = plates.joined()
        normalizedPlate = normalizeAndCleanPlate(normalizedPlate)
        
        if normalizedPlate.isEmpty {
            return (-1, "")
        }
        
        let plateCountryCode = stripCountryCode(normalizedPlate)
        
        
        if !plateCountryCode.isEmpty {
                if normalizedPlate.hasPrefix(plateCountryCode) {
                    normalizedPlate = String(normalizedPlate.dropFirst(plateCountryCode.count))
                }
        }
        
        
        if usePlatePattern{
            guard isValidPlate(normalizedPlate, plateCountryCode) else {
                return (-1, "")
            }
        }
       
        if normalizedPlate.count < MinSymbolInLP || normalizedPlate.count > MaxSymbolInLP {
            return (-1, "")
        }
        
        var newPlate: String
        if isStipCountryCode{
            newPlate =  normalizedPlate
        }
        else{
            newPlate =  plateCountryCode + normalizedPlate
        }
 
        var bestIndex: Int?
        //     var bestIoU: CGFloat = 0
        var bestScore: CGFloat = 0
        
        for (index, track) in tracks.enumerated() {
            //    if track.isLocked {
            //      let dist =  centerDistance(track.lastBox, box)
            let predictedBox = predictedRect(from: track.kalman!)
            let dist = centerDistance(predictedBox, box)
            
            if let last = track.plateBuffer.last {
                ocrScore = similarity(last, newPlate)
            } else {
                if !track.lockedPlate.isEmpty {
                    ocrScore = similarity(track.lockedPlate, newPlate)
                }
            }

            if dist < centerDistDebug  && ocrScore > ocrScoreThreshold {
                tracks[index].previousBox = tracks[index].lastBox
                //tracks[index].lastBox = smooth(old: tracks[index].previousBox, new: box)
                tracks[index].lastBox = smoothKalman(index: index, box: box)
                if !tracks[index].isLocked {
                    tracks[index].lastSeenFrame = frameIndex
                    tracks[index].lastSeenDate = Date()
                }
                tracks[index].lastBoxIndex = boxIndex
                tracks[index].missedFrames = 0
                if developerMode {
                    print("Matched 0 trackIndex:", index,  " centerDistance: ", dist, " lock: ", tracks[index].isLocked)
                }

                return (index,newPlate)
            }
            //   }
        }
        
        for (index, track) in tracks.enumerated() {
            //     let iouValue = iou(track.lastBox, box)
            //         let dist =  centerDistance(track.lastBox, box)
            let predictedBox = predictedRect(from: track.kalman!)
            let dist = centerDistance(predictedBox, box)
            
            let maxDist = max(box.width, box.height) * 2.0
            
            if dist > maxDist {
                continue // ❌ слишком далеко — не матчим
            }
            
            let iouValue = iou(predictedBox, box)
            
            if iouValue < 0.05 {
                continue
            }
            
            
            if let last = track.plateBuffer.last {
                ocrScore = similarity(last, newPlate)
            } else {
                if !track.lockedPlate.isEmpty {
                    ocrScore = similarity(track.lockedPlate, newPlate)
                }
                
            }
            
            let isMatch = ( iouValue > threshold
                            || dist < centerDistDebug
                            || ocrScore > ocrScoreThreshold )
            
            let score = iouValue * 0.5 +
            (1.0 / (1.0 + dist)) * 0.2 +
            ocrScore * 0.8
            
            if developerMode {
                print("Matched 1 trackIndex: \(index), iou: \(iouValue), centerDistance: \(dist), OCR Score: \(ocrScore),  score: \(score)")
            }
            
            if isMatch {
                //                let score = v  + (1.0/(1.0 + dist))
                if score > bestScore {
                    bestScore = score
                    bestIndex = index
                }
            }
            }
            
            if let index = bestIndex {
                tracks[index].previousBox = tracks[index].lastBox
                //tracks[index].lastBox = smooth(old: tracks[index].previousBox, new: box)
                tracks[index].lastBox = smoothKalman(index: index, box: box)
                if !tracks[index].isLocked {
                    tracks[index].lastSeenFrame = frameIndex
                    tracks[index].lastSeenDate = Date()
                }
                tracks[index].lastBoxIndex = boxIndex
                tracks[index].missedFrames = 0
                if developerMode {
                    print("Matched 2 trackIndex:", index,  " bestScore:", bestScore, " lock:", tracks[index].isLocked)
                }
                
                return (index, newPlate)
            }
        
        // ➕ новый трек
        var newTrack = PlateTrackMyself(
            bbox: box,
            frameIndex: frameIndex,
            boxIndex: boxIndex
        )
        newTrack.lastSeenDate = Date()
        
        let viewkIndex = self.viewIndex(for: newTrack.id) ?? -1
        tracks.append(newTrack)
        
        // find the newly appended track index (use closure syntax and return non-optional Int)
        let newIndex = tracks.firstIndex(where: { $0.id == newTrack.id }) ?? -1
        
        if developerMode {
            print("➕ New trackIndex: \(newIndex), viewIndex: \(viewkIndex)")
        }
        
        return (newIndex,newPlate)
    }
    
    //--------
    func predictedRect(from k: KalmanState) -> CGRect {
        let px = k.x + k.vx
        let py = k.y + k.vy
        
        return CGRect(
            x: px - k.w / 2,
            y: py - k.h / 2,
            width: k.w,
            height: k.h
        )
    }
    
    //-----
    func smoothKalman(index: Int, box: CGRect) -> CGRect {
        
        if var k = tracks[index].kalman {
            
            kalmanPredict(&k)
            let predictedX = k.x + k.vx
            let predictedY = k.y + k.vy
            
            let dist = hypot(predictedX - box.midX, predictedY - box.midY)
            
            if dist < 150 {
                kalmanUpdate(&k, measurement: box)
            }
             
            tracks[index].kalman = k
            
            let newBox = CGRect(
                x: k.x - k.w / 2,
                y: k.y - k.h / 2,
                width: k.w,
                height: k.h
            )
            
            return newBox
        }
        else {
            return box
        }
    }
    
    //---------
    func smooth(old: CGRect, new: CGRect) -> CGRect {
        // smoothingFactor должен быть CGFloat; используем напрямую
        let alpha = smoothingFactor
        // вычисляем промежуточные значения по осям
        let oldX = old.origin.x
        let oldY = old.origin.y
        let oldW = old.size.width
        let oldH = old.size.height
        let newX = new.origin.x
        let newY = new.origin.y
        let newW = new.size.width
        let newH = new.size.height
        
        let x = calcSmoothVariable(old: Float(oldX), new: Float(newX), smoothingFactor: alpha)
        let y = calcSmoothVariable(old: Float(oldY), new: Float(newY), smoothingFactor: alpha)
        let width = calcSmoothVariable(old: Float(oldW), new: Float(newW), smoothingFactor: alpha)
        let height = calcSmoothVariable(old: Float(oldH), new: Float(newH), smoothingFactor: alpha)
        
        return CGRect(x: x, y: y, width: width, height: height)
    }
    
    
    //
    func calcSmoothVariable(old: Float, new: Float, smoothingFactor: Float) -> CGFloat {
        return CGFloat(old * (1 - smoothingFactor) + new * smoothingFactor)
    }
    
    //Cleanup boutingBoxes
    func cleanupBoundingBoxesMyself() {
        boundingBoxViews.forEach {$0.hide()}
    }
    
    //
    func cleanupTracksMyself() {
        let now = Date()
        
        //   if developerMode {
        //       print("cleanupTracksMyself 0: unlockTimeout \(unlockTimeout) trackTTL: \(trackTTL) date now: \(now)")
        //   }
        
        
        var aliveTracks: [PlateTrackMyself] = []
        
        for track in tracks {
            let isAlive = now.timeIntervalSince(track.lastSeenDate) < unlockTimeout
            let isLossTime =  now.timeIntervalSince(track.lastSeenDate) < trackTTL
            
            /*     let viewIndex = trackToViewMap[track.id] ?? -1
             if developerMode {
             let trackIndex = self.tracks.firstIndex(where: {$0.id == track.id}) ?? -1
             print("cleanupTracksMyself 1: trackIndex \(trackIndex) viewIndex: \(viewIndex) is removed isLock: \(track.isLocked) isAlive: \(isAlive) isLossTime: \(isLossTime) track.lastSeenDate: \(track.lastSeenDate) timeIntervalSince: \(now.timeIntervalSince(track.lastSeenDate))" )
             }
             */
            if (isAlive && track.isLocked) || (isLossTime && !track.isLocked) {
                aliveTracks.append(track)
            }
            
            else {
                // 🔥 освобождаем view
                
                if let viewIndex = trackToViewMap[track.id] {
                    if developerMode {
                        let trackIndex = self.tracks.firstIndex(where: {$0.id == track.id}) ?? -1
                        print("cleanupTracksMyself 2: trackIndex \(trackIndex) viewIndex: \(viewIndex) is removed isLock: \(track.isLocked) isAlive: \(isAlive) isLossTime: \(isLossTime)" )
                    }
                    trackToViewMap.removeValue(forKey: track.id)
                    freeViewIndices.append(viewIndex)
                    boundingBoxViews[viewIndex].hide()
                }
            }
        }
        
        tracks = aliveTracks
    }
    
    
    
    //
    func removeAllTracks() {
        tracks.removeAll()
    }
    
    //
    func hideAllBoundingBoxes() {
        for i in 0..<self.boundingBoxViews.count {
            DispatchQueue.main.async {
                self.boundingBoxViews[i].hide()
            }
        }
    }
    
    
    //------Sound Feedback
    func playClickSound() {
        AudioServicesPlaySystemSound(1104) // Camera shutter-like click
    }
    
    //---------------
    func stripCountryCode(_ plate: String) -> String {
        for platePattern in platePatterns {
            let plateCode = platePattern.plateCode
            if plate.hasPrefix(plateCode),
               plate.count >= plateCode.count + MinSymbolInLP {
                return plateCode
            }
        }
        return ""
    }
    //------------
    func normalizePlate(_ text: String) -> String {
        return text.uppercased()
            .filter { $0.isLetter || $0.isNumber }
            .replacingOccurrences(of: " ", with: "")        
    }
    
    // MARK: - OCR Utils
    
    func isValidPlate(_ text: String, _ plateCountryCode: String) -> Bool {
        if developerMode {
            print("isValidPlate: plate: \(text), plateCountryCode: \(plateCountryCode) plateCountryParam: \(plateCountryParam)")
        }
        
        if  isStipCountryCode && !plateCountryCode.isEmpty {
            let matchingPatterns = platePatterns.filter { $0.plateCode == plateCountryCode }
            for platePattern in matchingPatterns {
                if text.range(of: platePattern.pattern, options: .regularExpression) != nil {
                    return true
                }
            }
        }
  
        if plateCountryParam == "AUTO" {
            for platePattern in platePatterns {
                if text.range(of: platePattern.pattern, options: .regularExpression) != nil {
                    return true
                }
            }
        }
        else {
            let matchingPatterns = platePatterns.filter { $0.isoCode == plateCountryParam}
            for platePattern in matchingPatterns {
                if text.range(of: platePattern.pattern, options: .regularExpression) != nil {
                    return true
                }
            }
       }
        
       return false
    }
    //--------
    func normalizeAndCleanPlate(_ text: String) -> String {
        let normalized = normalizePlate(text)
        return   normalized
        //return stripCountryCode(normalized)
    }
    
    //
    func processRecognizedPlates(_ plate: String,
                                 trackIndex: Int,
                                 normBoxCrop: CGRect,
                                 bestClass: String,
                                 alpha: CGFloat,
                                 indexBountingBox: Int) {
        guard tracks.indices.contains(trackIndex) else { return }
        
        tracks[trackIndex].plateBuffer.append(plate)
        
        if tracks[trackIndex].plateBuffer.count > maxBufferSize {
            tracks[trackIndex].plateBuffer.removeFirst(
                tracks[trackIndex].plateBuffer.count - maxBufferSize
            )
        }
        
        if developerMode {
            print("processRecognizedPlates 1: \(indexBountingBox) trackIndex: \(trackIndex) buffer:", tracks[trackIndex].plateBuffer)
        }
        
        tryLockPlate(
            trackIndex: trackIndex,
            normBoxCrop: normBoxCrop,
            bestClass: bestClass,
            alpha: alpha,
            indexBountingBox: indexBountingBox
        )
        
    }
    
    //-----------------
    func tryLockPlate(trackIndex: Int,
                      normBoxCrop: CGRect,
                      bestClass: String,
                      alpha: CGFloat,
                      indexBountingBox: Int) {
        
        guard tracks.indices.contains(trackIndex) else { return }
        
        if !tracks[trackIndex].isLocked  {

            let grouped = Dictionary(grouping: tracks[trackIndex].plateBuffer, by: { $0 })
            
            if developerMode {
                print("trackIndex: \(trackIndex) grouped:", grouped)
            }
            
            if let best = grouped.max(by: { $0.value.count < $1.value.count }),
               best.value.count >= lockThreshold {
                
                tracks[trackIndex].lockedPlate = best.key
                tracks[trackIndex].isLocked = true
                tracks[trackIndex].lockAt = Date()
                tracks[trackIndex].plateBuffer.removeAll()
                let trackCopy = tracks[trackIndex]
                
                DispatchQueue.main.async {
                    self.updateUIFunction(
                        track: trackCopy,
                        bestClass: bestClass,
                        alpha: alpha,
                        indexBountingBox: indexBountingBox
                    )
                    self.playClickSound()
                    self.playFeedbackGenerator()
                    
                }
            }
            else {
                tracks[trackIndex].lockedPlate = tracks[trackIndex].plateBuffer.last ?? ""
                let trackCopy = tracks[trackIndex]
                if !trackCopy.lockedPlate.isEmpty {
                    DispatchQueue.main.async {
                        self.updateUIFunction(
                            track: trackCopy,
                            bestClass: bestClass,
                            alpha: alpha,
                            indexBountingBox: indexBountingBox
                        )
                    }
                }
            }
        }
        else {
            let trackCopy = tracks[trackIndex]
            DispatchQueue.main.async {
                self.updateUIFunction(
                    track: trackCopy,
                    bestClass: bestClass,
                    alpha: alpha,
                    indexBountingBox: indexBountingBox
                )
            }
            
        }
    }
    
    //
    func viewIndex(for trackID: UUID) -> Int? {
        
        // уже есть назначенный view
        if let index = trackToViewMap[trackID] {
            return index
        }
        
        // берём свободный
        guard !freeViewIndices.isEmpty else {
            return nil
        }
        
        let index = freeViewIndices.removeFirst()
        trackToViewMap[trackID] = index
        
        return index
    }
    
    
    //
    func unlockAllTracks() {
        for i in tracks.indices {
            tracks[i].isLocked = false
            tracks[i].lockedPlate = ""
            tracks[i].plateBuffer.removeAll()
        }
    }
    
    
    //--
    func kalmanPredict(_ k: inout KalmanState) {
        k.x += k.vx
        k.y += k.vy
    }
    
    //------
    func kalmanUpdate(_ k: inout KalmanState, measurement: CGRect) {
        let mx = measurement.midX
        let my = measurement.midY
        
   //     let alpha: CGFloat = CGFloat(positionGain)  // доверие измерению
   //     let beta: CGFloat = CGFloat(velocityGain)    // скорость
        let dx = mx - k.x
        let dy = my - k.y
        let dist = sqrt(dx*dx + dy*dy)
        
        let alpha = min(0.75, max(0.45, 1 - dist / 200))
        let beta  = alpha * 0.3

        let newVx = mx - k.x
        let newVy = my - k.y
        
        k.vx = k.vx * (1 - beta) + newVx * beta
        k.vy = k.vy * (1 - beta) + newVy * beta
        
        k.x = k.x * (1 - alpha) + mx * alpha
        k.y = k.y * (1 - alpha) + my * alpha
        
        k.w = k.w * (1 - alpha) + measurement.width * alpha
        k.h = k.h * (1 - alpha) + measurement.height * alpha
        
    
    }
    
    //
    func pauseCamera() {
        guard videoCapture.captureSession.isRunning else { return }
   //     removeAllTracks()
        videoCapture.captureSession.stopRunning()
    }

    //
    func resumeCamera() {
        guard !videoCapture.captureSession.isRunning else { return }
        removeAllTracks()
        hideAllBoundingBoxes()
        freeViewIndices = Array(0..<boundingBoxViews.count)
        videoCapture.captureSession.startRunning()
    }

}
