//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  BoundingBoxView for Ultralytics YOLO App
//  This class is designed to visualize bounding boxes and labels for detected objects in the YOLOv8 models within the Ultralytics YOLO app.
//  It leverages Core Animation layers to draw the bounding boxes and text labels dynamically on the detection video feed.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  BoundingBoxView facilitates the clear representation of detection results, improving user interaction with the app by
//  providing immediate visual feedback on detected objects, including their classification and confidence level.

import Foundation
import UIKit

/// Manages the visualization of bounding boxes and associated labels for object detection results.
class BoundingBoxView: UIView {
  /// The layer that draws the bounding box around a detected object.
  let shapeLayer: CAShapeLayer

  /// The layer that displays the label and confidence score for the detected object.
  let textLayer: CATextLayer

  private var tappableTextRect: CGRect = .zero
    
  /// Последний отображённый фрейм (в координатах родительского слоя) — используется для hit-testing
  var lastFrame: CGRect? = nil

  /// Последняя отображённая строка метки
  var lastLabel: String? = nil

// private var tapRecognizer: UITapGestureRecognizer?
  private var tapAction: (() -> Void)?
  
  private lazy var recognizer = UITapGestureRecognizer(
        target: self,
        action: #selector(handleTap)
    )
    
  /// Initializes a new BoundingBoxView with configured shape and text layers.
  override init(frame: CGRect = .zero) {
    shapeLayer = CAShapeLayer()
    shapeLayer.fillColor = UIColor.clear.cgColor  // No fill to only show the bounding outline
    shapeLayer.lineWidth = 4  // Set the stroke line width
    shapeLayer.isHidden = true  // Initially hidden; shown when a detection occurs

    textLayer = CATextLayer()
    textLayer.isHidden = true  // Initially hidden; shown with label when a detection occurs
    textLayer.contentsScale = UIScreen.main.scale  // Ensure the text is sharp on retina displays
    textLayer.fontSize = 14  // Set font size for the label text
    textLayer.font = UIFont(name: "Avenir", size: textLayer.fontSize)  // Use Avenir font for labels
    textLayer.alignmentMode = .center  // Center-align the text within the layer

    super.init(frame: frame)
    self.addGestureRecognizer(recognizer)
    self.isUserInteractionEnabled = true
      
    self.clipsToBounds = false
    self.superview?.bringSubviewToFront(self)
      
    self.layer.addSublayer(shapeLayer)
    self.layer.addSublayer(textLayer)
  }

  required init?(coder: NSCoder) {
    fatalError("init(coder:) has not been implemented")
  }

  /// Updates the bounding box and label to be visible with specified properties.
  /// - Parameters:
  ///   - frame: The CGRect frame defining the bounding box's size and position.
  ///   - label: The text label to display (e.g., object class and confidence).
  ///   - color: The color of the bounding box stroke and label background.
  ///   - alpha: The opacity level for the bounding box stroke and label background.
  func show(frame: CGRect, label: String, color: UIColor, alpha: CGFloat, tapAction: (() -> Void)? = nil) {
    CATransaction.setDisableActions(true)  // Disable implicit animations
    self.frame = frame
      self.backgroundColor = color.withAlphaComponent(0.3)

    let path = UIBezierPath(roundedRect: self.bounds, cornerRadius: 6.0)  // Rounded rectangle for the bounding box
    shapeLayer.path = path.cgPath
    shapeLayer.strokeColor = color.withAlphaComponent(alpha).cgColor  // Apply color and alpha to the stroke
    shapeLayer.isHidden = false  // Make the shape layer visible

    textLayer.string = label  // Set the label text
    textLayer.backgroundColor = color.withAlphaComponent(alpha).cgColor  // Apply color and alpha to the background
    textLayer.isHidden = false  // Make the text layer visible
    textLayer.foregroundColor = UIColor.white.withAlphaComponent(alpha).cgColor  // Set text color
    textLayer.cornerRadius = 8
    textLayer.masksToBounds = true
      
    // Calculate the text size and position based on the label content
    let attributes = [NSAttributedString.Key.font: textLayer.font as Any]
    let textRect = label.boundingRect(
      with: CGSize(width: 400, height: 100),
      options: .usesLineFragmentOrigin,
      attributes: attributes, context: nil)
    let textSize = CGSize(width: textRect.width + 12, height: textRect.height)  // Add padding to the text size
   // let textOrigin = CGPoint(x: frame.origin.x - 2, y: frame.origin.y - textSize.height - 2)  // Position above the bounding box
    //textLayer.frame = CGRect(origin: textOrigin, size: textSize)  // Set the text layer frame
      
    textLayer.frame = CGRect(origin: CGPoint(x: 0, y: -textSize.height - 2), size: textSize)
 //My add
    tappableTextRect = textLayer.frame.insetBy(dx: -20, dy: -10)      // Save last values for hit-testing and retrieval
    
    self.lastFrame = frame
    self.lastLabel = label
 
      // Gesture logic
  //    if let old = tapRecognizer {
  //         self.removeGestureRecognizer(old)
  //     }

       // 📌 6. Добавляем новый, если есть действие
       if let tapAction = tapAction, !label.isEmpty {
           self.tapAction = tapAction
  //         let recognizer = UITapGestureRecognizer(target: self, action: #selector(handleTap))
  //         self.addGestureRecognizer(recognizer)
  //         tapRecognizer = recognizer
       } else {
           self.tapAction = nil
  //         self.tapRecognizer = nil
       }
  }

  //
  @objc private func handleTap() {
 //   print("✅ Tap detected on box with label: \(lastLabel ?? "none")")
    tapAction?()
  }

  /// Hides the bounding box and text layers.
  func hide() {
    shapeLayer.isHidden = true
    textLayer.isHidden = true
    lastFrame = nil
    lastLabel = nil
  //  tapRecognizer?.isEnabled = false
    tapAction = nil
    backgroundColor = .clear
  }

  /// Проверяет, содержит ли последний frame указанную точку (координаты в системе родительского слоя)
//  func contains(point: CGPoint) -> Bool {
//    guard let f = lastFrame else { return false }
 //   return f.contains(point)
//  }
    override func point(inside point: CGPoint, with event: UIEvent?) -> Bool {

        // Тап по тексту
        if tappableTextRect.contains(point) {
            return true
        }

        return bounds.insetBy(dx: -30, dy: -20).contains(point)
    }

//    override func point(inside point: CGPoint, with event: UIEvent?) -> Bool {
//        return bounds.contains(point)
//    }

    func contains(point: CGPoint) -> Bool {
        return bounds.contains(point)
    }
    
  var labelText: String? {
    return lastLabel
  }
}
