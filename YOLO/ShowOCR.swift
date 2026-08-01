//
//  ShoowOCR.swift
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

extension ViewController {
    //---------------------
    
    func show(predictions: [VNRecognizedObjectObservation]) {
        
        processEveryNFrames = Int(sliderOCR.value)
        
        self.labelSlider.text = String(predictions.count) + " items (max " + String(Int(slider.value)) + ")"
        self.labelOCR.text = "OCR every " + String(processEveryNFrames) + " frames"
        
        frameIndex += 1
        //cleanupBoundingBoxesMyself()
        cleanupTracksMyself()
        
        frameCounter += 1
        guard frameCounter >= (processEveryNFrames - 1) else { return }
        
       // cleanupTracksMyself()
        frameCounter = 0
        guard predictions.count > 0 else { return }
        
        
        let uiImage = currentBufferOCR?.toUIImage(orientation: .up)
        guard uiImage != nil  else { return }
        
        let prediction = predictions[0]
        let bestClass = prediction.labels.first?.identifier ?? ""
        let confidence = prediction.labels.first?.confidence ?? 0.0
        let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
        var boxes: [CGRect] = []
        
        var itemsCount: Int = 10
        if Thread.isMainThread {
            itemsCount = Int(slider.value)
        } else {
            DispatchQueue.main.sync {
                itemsCount = Int(slider.value)
            }
        }

        for i in 0..<predictions.count {
            let prediction = predictions[i]
            boxes.append(prediction.boundingBox)
        }
        
        let threshold = Double(round(100 * iOuTreshold)) / 100
        
        let filteredBoxes = suppressDuplicateBoxes(
            boxes: boxes,
            iouThreshold: threshold
        )
        let trackToViewMapCount = trackToViewMap.count

        if developerMode {
            print("✅ Start show() detected boxes: \(boxes.count), filtered boxes: \(filteredBoxes.count), trackCount: \(tracks.count)  itemsCount: \(itemsCount) trackToViewMapCount: \(trackToViewMapCount)")
        }
    

        let countLoop = min(filteredBoxes.count, itemsCount)

        guard countLoop > 0 else { return }

        for i in 0..<countLoop {
                guard let uiImage = uiImage else { return}
                let boundingBox = expandNormalizedBox(filteredBoxes[i],
                                                      by: CGFloat(padPercent_w),
                                                      by: CGFloat(padPercent_h))
                
                let normBox =   normalizedBoundingBoxForUIImage(
                        boundingBox: boundingBox,
                        videoPreview: videoPreview,
                    )
                
                let normBoxCrop = normBox
                if let crop = cropImage(uiImage, to: boundingBox) {
                    
                    if self.developerMode {
                        print("Show 1: i \(i) Crop image size: \(crop.size)")
                    }
                    var trackIndex: Int = 0
                    
                  //  let trackIndex = self.matchTrackMy(for: normBoxCrop, boxIndex: i)
                  //  let viewIndex = self.viewIndex(for: self.tracks[trackIndex].id) ?? -1
                  //  if self.developerMode {
                  //      print("Show 2: i \(i) trackIndex: \(trackIndex) viewIndex: \(viewIndex)")
                  //  }

                    recognizeText(from: crop) { texts in
                        if  !texts.isEmpty {
                            //let lastBoxIndex = self.tracks[trackIndex].lastBoxIndex
                            // match track (returns track index)
                            // trackIndex = self.matchTrackMy(for: normBoxCrop, boxIndex: i)
                            // if match failed, ignore
                            // guard trackIndex >= 0 && trackIndex < self.tracks.count else { return }
                            // let viewIndex = self.viewIndex(for: self.tracks[trackIndex].id) ?? -1
                            // match track (returns (trackIndex, newPlate))
                            let (matchedIndex, newPlate) = self.matchTrackMy(for: normBoxCrop, boxIndex: i, plates: texts)
                            trackIndex = matchedIndex
                            // if match failed, ignore
                            guard trackIndex >= 0 && trackIndex < self.tracks.count else { return }
                            let viewIndex = self.viewIndex(for: self.tracks[trackIndex].id) ?? -1
                             if self.developerMode {
                                 print("Show 2: i \(i) trackIndexCrop: \(trackIndex) viewIndex: \(viewIndex) texts: \(texts)")//  lastBoxIndex: \(lastBoxIndex)")
                             }
                             guard  trackIndex < self.tracks.count else { return }
                         //    if !self.tracks[trackIndex].isLocked {
                                 // DispatchQueue.main.async {
                                 self.processRecognizedPlates(
                                     newPlate,
                                     trackIndex: trackIndex,
                                     normBoxCrop: normBoxCrop,
                                     bestClass: bestClass,
                                     alpha: alpha,
                                     indexBountingBox: viewIndex
                                 )
                                 //}
                          //   }
                          //   else {
                               //  self.updateUIFunction(track: self.tracks[trackIndex], bestClass: bestClass, alpha: alpha, indexBountingBox: i)
                          //   }
                         }
                     }
                }
        }
        
         currentBufferOCR = nil
        
    }
        
        
        //Update UI with recognized text
        
    func updateUIFunction(track: PlateTrackMyself, bestClass:String, alpha:CGFloat, indexBountingBox: Int){
            
            DispatchQueue.main.async {
                let normBoxCrop = track.lastBox
                let recognizedLabel = track.lockedPlate
        
                let isLocked = track.isLocked
        
        //        let index = indexBountingBox
        //        guard index < self.boundingBoxViews.count else { return }
                
                guard let index = self.viewIndex(for: track.id),
                        self.boundingBoxViews.indices.contains(index) else {
                      return
                }
                            
                let boxColor: UIColor = isLocked
                ? UIColor.systemGreen      // 🟢 зеленая рамка
                : (self.colors[bestClass] ?? .white)
                
                if self.animateWithDuration > 0 {
                    UIView.animate(withDuration: self.animateWithDuration) {
                        self.boundingBoxViews[index].frame = normBoxCrop
                    }
                }
                
                self.boundingBoxViews[index].show(
                    frame: normBoxCrop,
                    label: recognizedLabel,
                    color: boxColor,
                    alpha: alpha,
                    tapAction: ( isLocked ? { [weak self] in
                            guard let self = self else { return }
                            let detectedVC = DetectedPlateViewController(plate: recognizedLabel)
                            detectedVC.delegate = self
                            self.pauseButton(nil)
                            detectedVC.modalPresentationStyle = .formSheet
                            self.present(detectedVC, animated: true)
                    } : nil
                    )
                 )
                }
            }
        
        
        //OCR Vision function
    func recognizeText(from image: UIImage, completion: @escaping ([String]) -> Void) {
            // Preprocess to improve OCR reliability
            let prepImage = preprocessPlateForOCR2(image)
            if  developerModeSaveImages {
                print("Save image after preprocessImageForOCR2")
                numFileCropImage += 1
                saveCropImage(prepImage, index: numFileCropImage)
            }
            guard let cg = prepImage.cgImage else {
                if developerMode {
                    print("[OCR] image has no cgImage")
                }
                return completion([])
            }
      //      if developerMode {
      //          print("[OCR] Vision OCR start (orig size: \(image.size), prep size: \(prepImage.size))")
      //      }
            let request = VNRecognizeTextRequest { request, error in
                if let error = error {
                    if self.developerMode {
                        print("[OCR] Vision error: \(error.localizedDescription)")
                    }
                }
                guard let observations = request.results as? [VNRecognizedTextObservation] else {
                    if self.developerMode {
                        print("[OCR] Vision returned no observations")
                    }
                    completion([])
                    return
                }
                let texts = observations.compactMap { $0.topCandidates(1).first?.string }
                completion(texts)
            }
            
            request.recognitionLevel = .accurate
            request.recognitionLanguages = [ "en-US", "en-GB" ]
            request.usesLanguageCorrection = false
            let handler = VNImageRequestHandler(cgImage: cg, options: [:])
            
            self.ocrQueue.async {
                self.ocrSemaphore.wait()
                defer { self.ocrSemaphore.signal() }

                try? handler.perform([request])
            }
            /*
            DispatchQueue.global(qos: .userInitiated).async {
                try? handler.perform([request])
            }
            */
        }
        
}
