//
//  MultiPlates.swift
//  YOLO
//
//  Created by Oleksandr Baibuz on 23.12.2025.
//  Copyright © 2025 Ultralytics. All rights reserved.
//

import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision


extension ViewController {
    //
    func normalize(_ text: String) -> String {
        text.uppercased()
            .replacingOccurrences(of: "[^A-Z0-9]", with: "", options: .regularExpression)
    }
    
    //
    func autocorrect(_ text: String) -> String {
        String(text.map { corrections[$0] ?? $0 })
    }
    
    //  Проверка соотвествия шаблону номеров
    func matchesPlatePattern(_ text: String) -> Bool {
   //     platePatterns.contains {
   //         NSPredicate(format: "SELF MATCHES %@", $0).evaluate(with: text)
    //    }
        return false
    }
    
    
    
    //Сопоставление YOLO → Track
    func matchTrack(for bbox: CGRect, index: Int) -> PlateTrack {
        //   let threshold: CGFloat = 0.4
        let threshold = Double(round(100 * iOuTreshold)) / 100
        
        if let track = plateTracks.values.first(where: {
            iou($0.bbox, bbox) > threshold
        }) {
            track.bbox = bbox
            track.lastSeen = Date()
            track.boundingBoxIndex = index
            return track
        }
        
        let newTrack = PlateTrack(bbox: bbox, index: index)
        plateTracks[newTrack.id] = newTrack
        return newTrack
    }
    
    //OCR → стабилизация
    func processOCR(_ text: String, for track: PlateTrack) {
        guard !track.isLocked else { return }
        
        let norm = autocorrect(normalize(text))
        if usePlatePattern {
            guard matchesPlatePattern(norm) else { return }
        }
        
        track.samples.append(norm)
        track.samples = Array(track.samples.suffix(10))
        tryLock(track)
        handleLockIfNeeded(track)
        
    }
    
    //
    func tryLock(_ track: PlateTrack) {
        let freq = Dictionary(grouping: track.samples, by: { $0 })
            .max { $0.value.count < $1.value.count }
        
        if let best = freq, best.value.count >= lockThreshold {
            track.isLocked = true
            track.lockedText = best.key
        }
      }

    //Очистка “пропавших” номеров
    func cleanupTracks() {
        let now = Date()
        plateTracks = plateTracks.filter {
            now.timeIntervalSince($0.value.lastSeen) < unlockTimeout
        }
    }
    
    //
    func drawTrack(_ track:PlateTrack, in boundingBox: BoundingBoxView){
        let color: UIColor = track.isLocked ? .systemGreen : .white
        boundingBox.show(
            frame: track.bbox,
            label: track.lockedText ?? "",
            color: color,
            alpha: 0.9,
            tapAction: track.isLocked ? { [weak self] in
                guard let self = self else { return }
                let detectedVC = DetectedPlateViewController(plate: track.lockedText ?? "")
                detectedVC.delegate = self
                self.pauseButton(nil)
                detectedVC.modalPresentationStyle = .formSheet
                self.present(detectedVC, animated: true)
            } : nil
        )
    }
    
    //
    func handleLockIfNeeded(_ track: PlateTrack) {
        guard track.isLocked else { return }
        guard !isCameraPaused else { return }

        DispatchQueue.main.async {
            self.drawTrack(track, in: self.boundingBoxViews[track.boundingBoxIndex])
            self.playFeedbackGenerator()
           //self.playClick()
        }
        
    //   isCameraPaused = true
    //    pauseCamera()
        
    //    autoResetCamera()
    }
    
    //
    func playClick() {
        AudioServicesPlaySystemSound(1104) // iOS camera click
    }
    
    //
    func playFeedbackGenerator() {
        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
    }
    
    //
    func autoResetCamera() {
        // Авто-сброс через N секунд
        DispatchQueue.main.asyncAfter(deadline: .now() + unlockTimeout) {
            self.isCameraPaused = false
            self.resumeCamera()
        }
    }
    
  }
