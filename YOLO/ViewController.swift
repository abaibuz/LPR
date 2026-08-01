//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  Main View Controller for Ultralytics YOLO App
//  This file is part of the Ultralytics YOLO app, enabling real-time object detection using YOLO11 models on iOS devices.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  This ViewController manages the app's main screen, handling video capture, model selection, detection visualization,
//  and user interactions. It sets up and controls the video preview layer, handles model switching via a segmented control,
//  manages UI elements like sliders for confidence and IoU thresholds, and displays detection results on the video feed.
//  It leverages CoreML, Vision, and AVFoundation frameworks to perform real-time object detection and to interface with
//  the device's camera.

import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision
import TesseractOCR

// Shared CIContext for image conversions
private let sharedCIContext = CIContext(options: nil)

var mlModel = try! lpr26m(configuration: .init()).model

class ViewController: UIViewController, AVCapturePhotoCaptureDelegate, VideoCaptureDelegate,DetectedPlateViewControllerDelegate {
    func detectedPlateDidClose() {
        isCameraPaused = false
        unlockAllTracks()
        playButton(self)
    }
    
    // Исправленная сигнатура метода делегата:
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
        predict(sampleBuffer: sampleBuffer)
    }
    
    @IBOutlet var videoPreview: UIView!
    @IBOutlet var View0: UIView!
    @IBOutlet var segmentedControl: UISegmentedControl!
    @IBOutlet var playButtonOutlet: UIBarButtonItem!
    @IBOutlet var pauseButtonOutlet: UIBarButtonItem!
    @IBOutlet var slider: UISlider!

     
    @IBOutlet weak var labelName: UILabel!
    @IBOutlet weak var labelFPS: UILabel!
    @IBOutlet weak var labelZoom: UILabel!
    @IBOutlet weak var labelVersion: UILabel!
    @IBOutlet weak var labelSlider: UILabel!
    
    
    
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    @IBOutlet weak var forcus: UIImageView!
    @IBOutlet weak var toolBar: UIToolbar!
    
    @IBOutlet weak var shareButtonOutlet: UIBarButtonItem!
    @IBOutlet weak var labelOCR: UILabel!
    @IBOutlet weak var sliderOCR: UISlider!
    
    let selection = UISelectionFeedbackGenerator()
    var detector = try! VNCoreMLModel(for: mlModel)
    var session: AVCaptureSession!
    var videoCapture: VideoCapture!
    var currentBuffer: CVPixelBuffer?
    var currentBufferOCR: CVPixelBuffer?
    var framesDone = 0
    var t0 = 0.0  // inference start
    var t1 = 0.0  // inference dt
    var t2 = 0.0  // inference dt smoothed
    var t3 = CACurrentMediaTime()  // FPS start
    var t4 = 0.0  // FPS dt smoothed
    // var cameraOutput: AVCapturePhotoOutput!
    var longSide: CGFloat = 3
    var shortSide: CGFloat = 4
    var frameSizeCaptured = false
    
    var ocrTimer: Timer?
    var frameCounter = 0
    var processEveryNFrames = 2 // ✅ OCR запускать каждые n кадров
    
    var numFileCropImage = 0
    
    var padPercent_w : Int {
        return UserDefaults.standard.object(forKey: "padPercent_w") as? Int ?? 10
    }    // 12 padding
    
    var padPercent_h : Int {
        return UserDefaults.standard.object(forKey: "padPercent_h") as? Int ?? 30
    }

    // Developer mode
    var developerMode : Bool = false
    //{
    //    return UserDefaults.standard.bool(forKey: "developerMode") as Bool
    //}   // developer mode selected in settings
    
    // Save Images in developer mode
    var developerModeSaveImages : Bool = false
    //{
    //    return UserDefaults.standard.bool(forKey: "developerModeSaveImages") as Bool
    //}   // developer mode selected in settings

    
    //
    var useTesseractOCR : Bool {
        return UserDefaults.standard.bool(forKey: "useTesseractOCR") as Bool
    }   // developer mode selected in settings

    var plateBuffer: [String] = []
    var maxBufferSize:  Int {
        return UserDefaults.standard.object(forKey: "bufferSize") as? Int ?? 8
    }
    var lockThreshold:  Int {
        return UserDefaults.standard.object(forKey: "lockThreshold") as? Int ?? 3
    }
    var usePlatePattern : Bool {
        return UserDefaults.standard.bool(forKey: "usePlatePattern") as Bool
    }   // developer mode selected in settings

    var isStipCountryCode : Bool {
        return UserDefaults.standard.bool(forKey: "isStipCountryCode") as Bool
    }   // developer mode selected in settings

    var centerDistanceThreshold : Int {
        return UserDefaults.standard.object(forKey: "centerDistanceThreshold") as? Int ?? 80
    }
 
    var plateCountryParam: String {
        return UserDefaults.standard.string(forKey: "plateCountry") ?? "AUTO"
    }
    
    var plateCountry: String = "AUTO"
    
    
    let save_detections = false  // write every detection to detections.txt
    let save_frames = false  // write every frame to frames.txt
  
    enum LPRState {
        case scanning
        case locked(String)
    }

    var lprState: LPRState = .scanning

//    private var plateBuffer: [String] = []

    
    let corrections: [Character: Character] = [
        "O": "0",
        "I": "1",
        "Z": "2",
        "S": "5",
        "B": "8"
    ]
    
    let countryCodes = [
    "AL","AND","A","BY","B","BIH","BG","HR","CY","CZ","DK",
    "EST","FIN","F","D","GB","GR","H","IS","IRL","I","LV",
    "LT","L","MK","M","MD","MC","MNE","NL","N","PL","P",
    "RO","RUS","SM","SRB","SK","SLO","E","S","CH","TR","UA","V"
    ]

    let platePatterns: [(isoCode: String, plateCode: String, pattern: String)] = [

        // 🇺🇦 Украина
        ("UA", "UA", "^[A-Z]{2}[0-9]{4}[A-Z]{2}$"),
        ("UA", "UA", "^[A-Z]{4}[0-9]{4}$"), // Квадратный номер
        ("UA", "UA", "^[0-9]{5}[A-Z]{2}"), // Старый номер
        
        // 🇩🇪 Германия
        ("DE", "D", "^[A-Z]{1,3}[A-Z]{1,2}[0-9]{1,4}$"),

        // 🇫🇷 Франция / 🇮🇹 Италия (одинаковый формат)
        ("FR", "F", "^[A-Z]{2}[0-9]{3}[A-Z]{2}$"),
        ("IT", "I", "^[A-Z]{2}[0-9]{3}[A-Z]{2}$"),

        // 🇬🇧 Великобритания
        ("UK", "GB", "^[A-Z]{2}[0-9]{2}[A-Z]{3}$"),

        // 🇪🇸 Испания
        ("ES", "E", "^[0-9]{4}[A-Z]{3}$"),

        // 🇳🇱 Нидерланды (несколько форматов)
        ("NL", "NL", "^[A-Z]{2}[0-9]{2}[A-Z]{2}$"),
        ("NL",  "NL", "^[0-9]{2}[A-Z]{2}[0-9]{2}$"),

        // 🇵🇱 Польша
        ("PL","PL", "^[A-Z]{2,3}[A-Z0-9]{4,5}$"),

        // 🇨🇿 Чехия
        ("CZ"," CZ", "^[0-9][A-Z][0-9]{4}$"),

        // 🇸🇪 Швеция
        ("SE", "S", "^[A-Z]{3}[0-9]{2}[A-Z0-9]$"),

        // 🇳🇴 Норвегия
        ("NO", "N", "^[A-Z]{2}[0-9]{5}$"),

        // 🇩🇰 Дания
        ("DK", "DK", "^[A-Z]{2}[0-9]{5}$"),

        // 🇫🇮 Финляндия
        ("FI", "FIN", "^[A-Z]{3}[0-9]{3}$"),

        // 🇧🇪 Бельгия
        ("BE", "B", "^[0-9]{1}[A-Z]{3}[0-9]{3}$"),

        // 🇦🇹 Австрия
        ("AT", "A", "^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,2}$"),

        // 🇨🇭 Швейцария
        ("CH", "CH", "^[A-Z]{2}[0-9]{1,6}$"),

        // 🇭🇺 Венгрия
        ("HU", "H", "^[A-Z]{3}[0-9]{3}$"),

        // 🇷🇴 Румыния
        ("RO", "RO", "^[A-Z]{1,2}[0-9]{2,3}[A-Z]{3}$"),

        // 🇧🇬 Болгария
        ("BG", "BG", "^[A-Z]{1,2}[0-9]{4}[A-Z]{2}$"),

        // 🇬🇷 Греция
        ("GR", "GR", "^[A-Z]{3}[0-9]{4}$"),

        // 🇵🇹 Португалия (несколько форматов)
        ("PT", "P", "^[A-Z]{2}[0-9]{2}[A-Z]{2}$"),
        ("PT", "P", "^[0-9]{2}[A-Z]{2}[0-9]{2}$"),

        // 🇮🇪 Ирландия
        ("IE",  "IRL", "^[0-9]{2}[A-Z]{1,2}[0-9]{1,6}$"),

        // 🇱🇹 Литва
        ("LT", "LT", "^[A-Z]{3}[0-9]{3}$"),

        // 🇱🇻 Латвия
        ("LV", "LV", "^[A-Z]{2}[0-9]{4}$"),

        // 🇪🇪 Эстония
        ("EE", "EST", "^[A-Z]{3}[0-9]{3}$"),

        // 🇺🇸 США (очень вариативно → слабый фильтр)
        ("US", "USA", "^[A-Z0-9]{5,8}$"),

        // 🇨🇦 Канада
        ("CA", "CDN", "^[A-Z]{3}[0-9]{3}$"),

        // 🌍 fallback (если ничего не подошло)
      //  ("UNKNOWN", "^[A-Z0-9]{5,8}$")
    ]
    
    //Класс для каждого номера
    final class PlateTrack {
        let id: UUID = UUID()
        var bbox: CGRect
        var samples: [String] = []
        var lastSeen: Date = Date()
        var isLocked = false
        var lockedText: String?
        var boundingBoxIndex: Int
        
        init(bbox: CGRect, index: Int) {
            self.bbox = bbox
            self.boundingBoxIndex = index
        }
     }
    
    var trackToViewMap: [UUID: Int] = [:]
    var freeViewIndices: [Int] = []
    
    //
    struct KalmanState {
        var x: CGFloat
        var y: CGFloat
        var vx: CGFloat
        var vy: CGFloat
        var w: CGFloat
        var h: CGFloat
    }
    
    //Класс для каждого номера
    struct PlateTrackMyself {
        let id: UUID = UUID()
        var previousBox: CGRect
        var lastBox: CGRect
        var plateBuffer: [String] = []
        var lockedPlate: String = ""
        var lastSeenFrame: Int = 0
        var lastSeenDate: Date = Date()
        
        var missedFrames: Int = 0          // 👈 NEW
        let maxMissedFrames: Int = 5        // 👈 можно менять
        var elapsed: TimeInterval = 0.0
        
        var lastBoxIndex: Int = 0
        var isLocked = false
        var lockAt: Date = Date()
        
        var kalman: KalmanState?
        
        init(bbox: CGRect, frameIndex: Int,  boxIndex: Int) {
            self.previousBox = bbox
            self.lastBox = bbox
            self.lastSeenFrame = frameIndex
            self.lastBoxIndex = boxIndex
            
            self.kalman = KalmanState(
                x: bbox.midX,
                y: bbox.midY,
                vx: 0,
                vy: 0,
                w: bbox.width,
                h: bbox.height
            )
        }
 
        
    }


    var tracks: [PlateTrackMyself] = []
    var frameIndex: Int = 0
    // сколько кадров живёт трек без обновления
    
    var trackTTL  : TimeInterval {
        let ms = UserDefaults.standard.integer(forKey: "trackTTL")
        return ms > 0 ? TimeInterval(ms) / 1000.0 : 0.5
    }
    

    //
    var isCameraPaused: Bool = false
    
    var plateTracks: [UUID: PlateTrack] = [:]

//
    var unlockTimeout: TimeInterval {
        let ms = UserDefaults.standard.integer(forKey: "unlockTimeout")
        return ms > 0 ? TimeInterval(ms) / 1000.0 : 2.0
    }
    
    var smoothingFactor: Float {
        let ss = UserDefaults.standard.float(forKey: "smoothingFactor")
        if ss.isNaN || ss < 0.0 || ss > 1.0 {
            return 0.2
        }
        return ss
    }
    
    var positionGain: Float {
        let ss = UserDefaults.standard.float(forKey: "positionGain")
        if ss.isNaN || ss < 0.0 || ss > 1.0 {
            return 0.7
        }
        return ss
    }
 
    var velocityGain: Float {
        let ss = UserDefaults.standard.float(forKey: "velocityGain")
        if ss.isNaN || ss < 0.0 || ss > 1.0 {
            return 0.3
        }
        return ss
    }
    
    var iOuTreshold: Float {
        let ss = UserDefaults.standard.float(forKey: "iOuTreshold")
        if ss.isNaN || ss < 0.0 || ss > 1.0 {
            return 0.45
        }
        return ss
    }

    
    var ocrThreshold: Float {
        let ss = UserDefaults.standard.float(forKey: "ocrThreshold")
        if ss.isNaN || ss < 0.0 || ss > 1.0 {
            return 0.6
        }
        return ss
    }
    
    var animateWithDuration: TimeInterval {
        let ms = UserDefaults.standard.integer(forKey: "animateWithDuration")
        return ms > 0 ? TimeInterval(ms) / 1000.0 : 0.0
    }

    let ocrQueue = DispatchQueue(label: "ocr.queue")
    let ocrSemaphore = DispatchSemaphore(value: 1)
    
    
    //
    lazy var visionRequest: VNCoreMLRequest = {
        let request = VNCoreMLRequest(
            model: detector,
            completionHandler: {
                [weak self] request, error in
                self?.processObservations(for: request, error: error)
            })
        // NOTE: BoundingBoxView object scaling depends on request.imageCropAndScaleOption https://developer.apple.com/documentation/vision/vnimagecropandscaleoption
        request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
        return request
    }()
    
    //

    // Обновляет labelOCR в зависимости от текущего OCR
    func updateOCRLabel() {
     //   let ocrType = useTesseractOCR ? "Tesseract" : "Vision"
     //   labelOCR.text = "OCR: \(ocrType)"
    }

    
    override func viewDidLoad() {
        super.viewDidLoad()
        slider.value = 5
        sliderOCR.value = Float(processEveryNFrames)
        //   setTimer()
        setLabels()
        setUpBoundingBoxViews()
        setUpOrientationChangeNotification()
        startVideo()
        // labelVersion скрыта
        labelVersion.isHidden = true
    //    ocrSwitch.isOn = useTesseractOCR
    //    updateOCRLabel() // Обновить индикатор OCR при запуске
        // setModel()
    }
    
    private let cameraSessionQueue = DispatchQueue(
        label: "camera.session.queue"
    )
    
    
    
    override func viewWillTransition(
        to size: CGSize, with coordinator: any UIViewControllerTransitionCoordinator
    ) {
        super.viewWillTransition(to: size, with: coordinator)
        removeAllTracks()
        hideAllBoundingBoxes()
        freeViewIndices = Array(0..<boundingBoxViews.count)
        
        cameraSessionQueue.async {
            self.videoCapture.captureSession.startRunning()
        }
        
        if size.width > size.height {
           toolBar.setBackgroundImage(UIImage(), forToolbarPosition: .any, barMetrics: .default)
           toolBar.setShadowImage(UIImage(), forToolbarPosition: .any)
            
        } else {
             toolBar.setBackgroundImage(nil, forToolbarPosition: .any, barMetrics: .default)
            toolBar.setShadowImage(nil, forToolbarPosition: .any)
        }
        self.videoCapture.previewLayer?.frame = CGRect(
            x: 0, y: 0, width: size.width, height: size.height)
      //  print("viewWillTransition  \(size)")
       // self.videoCapture.previewLayer?.frame = self.videoPreview.bounds
    }
       
    
    
    //-------
    private func setTimer() {
        ocrTimer = Timer.scheduledTimer(timeInterval: 2.0,
                                        target: self,
                                        selector: #selector(runYOLOAndOCR),
                                        userInfo: nil,
                                        repeats: true)
    }
    
    private func setUpOrientationChangeNotification() {
        NotificationCenter.default.addObserver(
            self, selector: #selector(orientationDidChange),
            name: UIDevice.orientationDidChangeNotification, object: nil)
    }
    
    @objc func orientationDidChange() {
        videoCapture.updateVideoOrientation()
        //      frameSizeCaptured = false
    }
    
    @IBAction func vibrate(_ sender: Any) {
        selection.selectionChanged()
    }
    
    @IBAction func indexChanged(_ sender: Any) {
        selection.selectionChanged()
        activityIndicator.startAnimating()
        
        /// Switch model
        switch segmentedControl.selectedSegmentIndex {
        case 0:
            self.labelName.text = "LPR26n"
            mlModel = try! lpr26n_8(configuration: .init()).model
        case 1:
            self.labelName.text = "LPR26s"
            mlModel = try! lpr26s(configuration: .init()).model
        case 2:
            self.labelName.text = "LPR26m"
            mlModel = try! lpr26m(configuration: .init()).model
        case 3:
            self.labelName.text = "LPR26l"
            mlModel = try! lpr26l(configuration: .init()).model
        case 4:
            self.labelName.text = "LPR26x"
            mlModel = try! lpr26x(configuration: .init()).model
        default:
            break
        }
        setModel()
        setUpBoundingBoxViews()
        activityIndicator.stopAnimating()
    }
    
    func setModel() {
        
        /// VNCoreMLModel
        detector = try! VNCoreMLModel(for: mlModel)
        detector.featureProvider = ThresholdProvider()
        
        /// VNCoreMLRequest
        let request = VNCoreMLRequest(
            model: detector,
            completionHandler: { [weak self] request, error in
                self?.processObservations(for: request, error: error)
            })
        request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
        visionRequest = request
        t2 = 0.0  // inference dt smoothed
        t3 = CACurrentMediaTime()  // FPS start
        t4 = 0.0  // FPS dt smoothed
    }
    
    /// Update thresholds from slider values
    @IBAction func sliderChanged(_ sender: Any) {
        let conf = Double(round(100 * 0.25)) / 100
        detector.featureProvider = ThresholdProvider(iouThreshold: Double(iOuTreshold), confidenceThreshold: conf)
    }
    
    @IBAction func takePhoto(_ sender: Any?) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        
        // 1. captureSession and cameraOutput
        // session = videoCapture.captureSession  // session = AVCaptureSession()
        // session.sessionPreset = AVCaptureSession.Preset.photo
        // cameraOutput = AVCapturePhotoOutput()
        // cameraOutput.isHighResolutionCaptureEnabled = true
        // cameraOutput.isDualCameraDualPhotoDeliveryEnabled = true
        // print("1 Done: ", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1E9)
        
        // 2. Settings
        let settings = AVCapturePhotoSettings()
        // settings.flashMode = .off
        // settings.isHighResolutionPhotoEnabled = cameraOutput.isHighResolutionCaptureEnabled
        // settings.isDualCameraDualPhotoDeliveryEnabled = self.videoCapture.cameraOutput.isDualCameraDualPhotoDeliveryEnabled
        
        // 3. Capture Photo
        usleep(20_000)  // short 10 ms delay to allow camera to focus
        self.videoCapture.cameraOutput.capturePhoto(
            with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
        print("3 Done: ", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1E9)
    }
    
    @IBAction func logoButton(_ sender: Any) {
        selection.selectionChanged()
        if let link = URL(string: "https://www.ultralytics.com") {
            UIApplication.shared.open(link)
        }
    }
    
    func setLabels() {
        self.labelName.text = "LPR26m"
        self.labelVersion.text = "Version " + UserDefaults.standard.string(forKey: "app_version")!
    }
    
    @IBAction func playButton(_ sender: Any) {
        selection.selectionChanged()
        self.videoCapture.start()
        playButtonOutlet.isEnabled = false
        pauseButtonOutlet.isEnabled = true
        shareButtonOutlet.isEnabled = true
    }
    
    @IBAction func pauseButton(_ sender: Any?) {
        selection.selectionChanged()
        self.removeAllTracks()
        self.cleanupBoundingBoxesMyself()
        self.videoCapture.stop()
        playButtonOutlet.isEnabled = true
        pauseButtonOutlet.isEnabled = false
        shareButtonOutlet.isEnabled = false
    }
    
    @IBAction func switchCameraTapped(_ sender: Any) {
        self.videoCapture.captureSession.beginConfiguration()
        let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput
        self.videoCapture.captureSession.removeInput(currentInput!)
        // let newCameraDevice = currentInput?.device == .builtInWideAngleCamera ? getCamera(with: .front) : getCamera(with: .back)
        
        let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)!
        guard let videoInput1 = try? AVCaptureDeviceInput(device: device) else {
            return
        }
        
        self.videoCapture.captureSession.addInput(videoInput1)
        self.videoCapture.captureSession.commitConfiguration()
    }
    
    // share image
    @IBAction func shareButton(_ sender: Any) {
        selection.selectionChanged()
        let settings = AVCapturePhotoSettings()
  //      if #available(iOS 16.0, *) {
            // Use the largest dimensions the device supports
  //          settings.maxPhotoDimensions = .init(width: .max, height: .max)
  //      } else {
   //         settings.isHighResolutionPhotoEnabled = true
  //      }
        self.videoCapture.cameraOutput.capturePhoto(
            with: settings, delegate: self)
    }
    
    // share screenshot
    @IBAction func saveScreenshotButton(_ shouldSave: Bool = true) {
        // let layer = UIApplication.shared.keyWindow!.layer
        // let scale = UIScreen.main.scale
        // UIGraphicsBeginImageContextWithOptions(layer.frame.size, false, scale);
        // layer.render(in: UIGraphicsGetCurrentContext()!)
        // let screenshot = UIGraphicsGetImageFromCurrentImageContext()
        // UIGraphicsEndImageContext()
        
        // let screenshot = UIApplication.shared.screenShot
        // UIImageWriteToSavedPhotosAlbum(screenshot!, nil, nil, nil)
    }
    
    let maxBoundingBoxViews = 20
    var boundingBoxViews = [BoundingBoxView]()
    var colors: [String: UIColor] = [:]
    let MinBoxHeight: CGFloat = 0.02
    let MinBoxWidth: CGFloat = 0.05
//    let MinSymbolInLP = UserDefaults.standard.integer(forKey: "MinSymbolInLP")
//    let MaxSymbolInLP = UserDefaults.standard.integer(forKey: "MaxSymbolInLP")
    var MinSymbolInLP: Int {
        UserDefaults.standard.object(forKey: "MinSymbolInLP") as? Int ?? 4
    }

    var MaxSymbolInLP: Int {
        UserDefaults.standard.object(forKey: "MaxSymbolInLP") as? Int ?? 10
    }

    // Overlay views for tap handling (one per bounding box)
    // var overlayTapViews: [UIView] = []
    // Store last recognized label for each box
    // var currentRecognizedLabels: [String] = []
     let ultralyticsColorsolors: [UIColor] = [
        UIColor(red: 4 / 255, green: 42 / 255, blue: 255 / 255, alpha: 0.6),  // #042AFF
        UIColor(red: 11 / 255, green: 219 / 255, blue: 235 / 255, alpha: 0.6),  // #0BDBEB
        UIColor(red: 243 / 255, green: 243 / 255, blue: 243 / 255, alpha: 0.6),  // #F3F3F3
        UIColor(red: 0 / 255, green: 223 / 255, blue: 183 / 255, alpha: 0.6),  // #00DFB7
        UIColor(red: 17 / 255, green: 31 / 255, blue: 104 / 255, alpha: 0.6),  // #111F68
        UIColor(red: 255 / 255, green: 111 / 255, blue: 221 / 255, alpha: 0.6),  // #FF6FDD
        UIColor(red: 255 / 255, green: 68 / 255, blue: 79 / 255, alpha: 0.6),  // #FF444F
        UIColor(red: 204 / 255, green: 237 / 255, blue: 0 / 255, alpha: 0.6),  // #CCED00
        UIColor(red: 0 / 255, green: 243 / 255, blue: 68 / 255, alpha: 0.6),  // #00F344
        UIColor(red: 189 / 255, green: 0 / 255, blue: 255 / 255, alpha: 0.6),  // #BD00FF
        UIColor(red: 0 / 255, green: 180 / 255, blue: 255 / 255, alpha: 0.6),  // #00B4FF
        UIColor(red: 221 / 255, green: 0 / 255, blue: 186 / 255, alpha: 0.6),  // #DD00BA
        UIColor(red: 0 / 255, green: 255 / 255, blue: 255 / 255, alpha: 0.6),  // #00FFFF
        UIColor(red: 38 / 255, green: 192 / 255, blue: 0 / 255, alpha: 0.6),  // #26C000
        UIColor(red: 1 / 255, green: 255 / 255, blue: 179 / 255, alpha: 0.6),  // #01FFB3
        UIColor(red: 125 / 255, green: 36 / 255, blue: 255 / 255, alpha: 0.6),  // #7D24FF
        UIColor(red: 123 / 255, green: 0 / 255, blue: 104 / 255, alpha: 0.6),  // #7B0068
        UIColor(red: 255 / 255, green: 27 / 255, blue: 108 / 255, alpha: 0.6),  // #FF1B6C
        UIColor(red: 252 / 255, green: 109 / 255, blue: 47 / 255, alpha: 0.6),  // #FC6D2F
        UIColor(red: 162 / 255, green: 255 / 255, blue: 11 / 255, alpha: 0.6),  // #A2FF0B
    ]
    
    func setUpBoundingBoxViews() {
        // Ensure all bounding box views are initialized up to the maximum allowed.
        while boundingBoxViews.count < maxBoundingBoxViews {
            let boxView = BoundingBoxView()
            boxView.isUserInteractionEnabled = true
     //       let tap = UITapGestureRecognizer(target: self, action: #selector(boundingBoxTapped(_:)))
      //      boxView.addGestureRecognizer(tap)
            boundingBoxViews.append(boxView)
        }
 
        freeViewIndices = Array(0..<boundingBoxViews.count)
        
         // Retrieve class labels directly from the CoreML model's class labels, if available.
         guard let classLabels = mlModel.modelDescription.classLabels as? [String] else {
             fatalError("Class labels are missing from the model description")
         }

         // Assign colors to the classes.
         var count = 0
         for label in classLabels {
             let color = ultralyticsColorsolors[count]
             count += 1
             if count > ultralyticsColorsolors.count - 1 {
                 count = 0
             }
             colors[label] = color
         }
     }
     
     func startVideo() {
         videoCapture = VideoCapture()
         videoCapture.delegate = self
         self.videoPreview.isUserInteractionEnabled = true
         
         videoCapture.setUp(sessionPreset: .photo) { success in
             // .hd4K3840x2160 or .photo (4032x3024)  Warning: 4k may not work on all devices i.e. 2019 iPod
             if success {
                 // Add the video preview into the UI.
                 if let previewLayer = self.videoCapture.previewLayer {
                     self.videoPreview.layer.addSublayer(previewLayer)
                     self.videoCapture.previewLayer?.frame = self.videoPreview.bounds  // resize preview layer
                 }
                 
                 // Add the bounding box layers to the UI, on top of the video preview.
                 for box in self.boundingBoxViews {
                     self.videoPreview.addSubview(box)
                     self.videoPreview.bringSubviewToFront(box)
                 }

                 // Once everything is set up, we can start capturing live video.
                 self.videoCapture.start()
             }
         }
     }
    
    func predict(sampleBuffer: CMSampleBuffer) {
        if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            currentBuffer = pixelBuffer
            if !frameSizeCaptured {
                let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                longSide = max(frameWidth, frameHeight)
                shortSide = min(frameWidth, frameHeight)
                frameSizeCaptured = true
            }
            /// - Tag: MappingOrientation
            // The frame is always oriented based on the camera sensor,
            // so in most cases Vision needs to rotate it for the model to work as expected.
            let imageOrientation: CGImagePropertyOrientation
            switch UIDevice.current.orientation {
            case .portrait:
                imageOrientation = .up
            case .portraitUpsideDown:
                imageOrientation = .down
            case .landscapeLeft:
                imageOrientation = .up
            case .landscapeRight:
                imageOrientation = .up
            case .unknown:
                imageOrientation = .up
            default:
                imageOrientation = .up
            }
            
            // Invoke a VNRequestHandler with that image
            let handler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
            if UIDevice.current.orientation != .faceUp {  // stop if placed down on a table
                t0 = CACurrentMediaTime()  // inference start
                do {
                    try handler.perform([visionRequest])
                } catch {
                    print(error)
                }
                t1 = CACurrentMediaTime() - t0  // inference dt
            }
            currentBufferOCR = currentBuffer
            currentBuffer = nil
        }
    }
    
    func processObservations(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            if let results = request.results as? [VNRecognizedObjectObservation] {
                self.show(predictions: results)
            } else {
                self.show(predictions: [])
            }
            
            // Measure FPS
            if self.t1 < 10.0 {  // valid dt
                self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
            }
            self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
            self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)  // t2 seconds to ms
            self.t3 = CACurrentMediaTime()
        }
    }
    
    // Save text file
    func saveText(text: String, file: String = "saved.txt") {
        if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let fileURL = dir.appendingPathComponent(file)
            
            // Writing
            do {  // Append to file if it exists
                let fileHandle = try FileHandle(forWritingTo: fileURL)
                fileHandle.seekToEndOfFile()
                fileHandle.write(text.data(using: .utf8)!)
                fileHandle.closeFile()
            } catch {  // Create new file and write
                do {
                    try text.write(to: fileURL, atomically: false, encoding: .utf8)
                } catch {
                    print("no file written")
                }
            }
            
            // Reading
            // do {let text2 = try String(contentsOf: fileURL, encoding: .utf8)} catch {/* error handling here */}
        }
    }
    
    // Save image file
    func saveImage() {
        let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        let fileURL = dir!.appendingPathComponent("saved.jpg")
        let image = UIImage(named: "ultralytics_yolo_logotype.png")
        FileManager.default.createFile(
            atPath: fileURL.path, contents: image!.jpegData(compressionQuality: 0.5), attributes: nil)
    }
    
    // Return hard drive space (GB)
    func freeSpace() -> Double {
        let fileURL = URL(fileURLWithPath: NSHomeDirectory() as String)
        do {
            let values = try fileURL.resourceValues(forKeys: [
                .volumeAvailableCapacityForImportantUsageKey
            ])
            return Double(values.volumeAvailableCapacityForImportantUsage!) / 1E9  // Bytes to GB
        } catch {
            print("Error retrieving storage capacity: \(error.localizedDescription)")
        }
        return 0
    }
    
    // Return RAM usage (GB)
    func memoryUsage() -> Double {
        var taskInfo = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let kerr: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
            $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        if kerr == KERN_SUCCESS {
            return Double(taskInfo.resident_size) / 1E9  // Bytes to GB
        } else {
            return 0
        }
    }
    
    // ✅ YOLO + OCR по таймеру
    @objc func runYOLOAndOCR() {
        //if let uiImage = currentBufferOCR?.toUIImage(orientation: .right) {
          //  currentBufferOCR = nil
            /*             if let crop = cropImage(uiImage, to: rect_ocr) {
             print("rect_ocr =\(rect_ocr)")
             recognizeText(from: crop) { texts in
             print("LP =\(texts)")
             label = texts.joined(separator: " ")
             self.boundingBoxViews[i].show(
             frame: rect,
             label: label,
             color: self.colors[bestClass] ?? UIColor.white,
             alpha: alpha)  // alpha 0 (transparent) to 1 (opaque) for conf threshold 0.2 to 1.0)
             }
             }
             */
       // }
    }
  
    //----
    /*
  
    func recognizeTextWithTesseract(image: UIImage, completion: @escaping ([String]) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            var result: [String] = []
            // Диагностика наличия файла eng.traineddata
            if Bundle.main.path(forResource: "tessdata/eng", ofType: "traineddata") != nil {
               // print("[OCR] Tesseract start (image size: \(image.size))")
                // Preprocess image for Tesseract
                let prep = self.preprocessImageForOCR(image)
                if let tesseract = G8Tesseract(language: "eng") {
                    tesseract.engineMode = .tesseractOnly
                    // For license plates prefer single-line / single-block mode
                    tesseract.pageSegmentationMode = .singleLine
                    // Limit characters to uppercase letters and digits to improve accuracy
                    tesseract.setVariableValue("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", forKey: "tessedit_char_whitelist")
                    tesseract.image = prep
                    tesseract.recognize()
                    if let text = tesseract.recognizedText?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
                        result = text.components(separatedBy: "\n").filter { !$0.isEmpty }
                    }
                 //   print("[OCR] Tesseract recognized: \(result)")
                } else {
                 //   print("[OCR] Tesseract init failed")
                }
            } else {
                print("eng.traineddata НЕ найден в бандле!")
            }
             DispatchQueue.main.async {
                 completion(result)
             }
         }
    }
     */
    func recognizeTextWithTesseract(
        image: UIImage,
        completion: @escaping ([String]) -> Void
    ) {
        DispatchQueue.global(qos: .userInitiated).async {

            var results: [String] = []

            guard Bundle.main.path(forResource: "tessdata/eng", ofType: "traineddata") != nil else {
                print("[OCR] eng.traineddata not found")
                DispatchQueue.main.async { completion([]) }
                return
            }

            let prepImage = self.preprocessPlateForOCR2(image)

            guard let tesseract = G8Tesseract(language: "eng") else {
                DispatchQueue.main.async { completion([]) }
                return
            }

            // 🔥 Оптимальные настройки для номерных знаков
            tesseract.engineMode = .tesseractOnly
            tesseract.pageSegmentationMode = .auto//.singleLine

            // 🔒 Ограничиваем символы
            tesseract.setVariableValue(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                forKey: "tessedit_char_whitelist"
            )

            // ❌ Отключаем словари
            tesseract.setVariableValue("0", forKey: "load_system_dawg")
            tesseract.setVariableValue("0", forKey: "load_freq_dawg")

            tesseract.image = prepImage
            tesseract.recognize()

            if let rawText = tesseract.recognizedText {
                let cleaned = rawText
                    .uppercased()
                    .components(separatedBy: .newlines)
                    .map { String($0.filter { $0.isLetter || $0.isNumber }) }
                    .filter {
                        !$0.isEmpty &&
                        $0.count >= self.MinSymbolInLP &&
                        $0.count <= self.MaxSymbolInLP
                    }

                results = cleaned
            }

            DispatchQueue.main.async {
                completion(results)
            }
        }
    }
    //--------------
    func cropImage(_ img: UIImage, to rect: CGRect) -> UIImage? {
        // This function accepts EITHER:
        // - a Vision-normalized rect (values in 0..1, origin = bottom-left), OR
        // - a rect in view coordinates (pixels relative to videoPreview) where values are > 1.
        // We'll detect which case by checking rect max dimension.

        guard let cgImage = img.cgImage else {
     //       print("[cropImage] no cgImage available")
            return nil
        }
        let pixelWidth = CGFloat(cgImage.width)
        let pixelHeight = CGFloat(cgImage.height)

        var normRect = rect
        // Heuristic: if rect coordinate values exceed 1 (e.g., in pixels of the view), treat as view-space rect
        if rect.maxX > 1.0 || rect.maxY > 1.0 {
            // rect is in view coordinates (videoPreview). Convert to normalized coords relative to the image.
            let displayRect = imageDisplayRectInView(imageSize: img.size, viewSize: videoPreview.bounds.size)

            // Compute intersection with displayRect to avoid negative coords
            let intersectRect = rect.intersection(displayRect)
            if intersectRect.isNull || intersectRect.width <= 0 || intersectRect.height <= 0 {
      //          print("[cropImage] input view-rect does not intersect displayed image area")
                return nil
            }

            // Convert from displayRect pixels to normalized image coordinates (origin bottom-left)
            let xInDisplay = intersectRect.origin.x - displayRect.origin.x
            let yInDisplay = intersectRect.origin.y - displayRect.origin.y

            let nx = xInDisplay / displayRect.width
            // Vision normalized origin is bottom-left; display coords origin is top-left, so compute bottom-left y
            let ny = 1.0 - ((yInDisplay + intersectRect.height) / displayRect.height)
            let nw = intersectRect.width / displayRect.width
            let nh = intersectRect.height / displayRect.height

            normRect = CGRect(x: nx, y: ny, width: nw, height: nh)
     //      print("[cropImage] converted viewRect=\(rect) -> intersect=\(intersectRect) -> normRect=\(normRect)")
        } else {
            // Assume rect is already normalized (Vision-style: origin bottom-left)
            normRect = rect
        }

        // Now convert normalized rect -> CGImage pixel coords
        let px = normRect.origin.x * pixelWidth
        let pWidth = normRect.size.width * pixelWidth
        let py = (1.0 - normRect.origin.y - normRect.size.height) * pixelHeight
        let pHeight = normRect.size.height * pixelHeight

        var pixelRect = CGRect(x: floor(px), y: floor(py), width: ceil(pWidth), height: ceil(pHeight))
        let imageBounds = CGRect(x: 0, y: 0, width: pixelWidth, height: pixelHeight)
        pixelRect = pixelRect.intersection(imageBounds)

        // Debug logging
  //      print("[cropImage] img.size=(\(img.size.width)x\(img.size.height)), cgSize=(\(Int(pixelWidth))x\(Int(pixelHeight)))")
  //      print("[cropImage] normRect=\(normRect) -> pixelRect=\(pixelRect)")

        if pixelRect.width <= 0 || pixelRect.height <= 0 {
   //         print("[cropImage] pixelRect has non-positive area, skipping crop")
            return nil
        }
        pixelRect = expandNormalizedBox(pixelRect, by: CGFloat(padPercent_w), by: CGFloat(padPercent_h))
        
        guard let croppedCg = cgImage.cropping(to: pixelRect) else {
    //        print("[cropImage] cgImage.cropping returned nil")
            return nil
        }

        let croppedImg = UIImage(cgImage: croppedCg, scale: img.scale, orientation: img.imageOrientation)
        return croppedImg
    }
    
    //--------------
    func saveCropImage(_ image: UIImage, index: Int) {
        if let data = image.jpegData(compressionQuality: 1.0) {
            let filename = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("crop_\(index).jpg")
            try? data.write(to: filename)
            print("Crop saved: \(filename)")
        }
    }
    
    // Вспомогательная функция для визуализации crop-области поверх видео
    func showCropRect(_ rect: CGRect, color: UIColor = .red) {
        // Удаляем старые crop-rect
        videoPreview.layer.sublayers?.filter { $0.name == "CropRectLayer" }.forEach { $0.removeFromSuperlayer() }
        let cropLayer = CAShapeLayer()
        cropLayer.name = "CropRectLayer"
        cropLayer.frame = videoPreview.bounds
        let path = UIBezierPath(rect: rect)
        cropLayer.path = path.cgPath
        cropLayer.strokeColor = color.cgColor
        cropLayer.lineWidth = 2.0
        cropLayer.fillColor = UIColor.clear.cgColor
        videoPreview.layer.addSublayer(cropLayer)
    }
    
 
    // Expand normalized rect by percent (0..1) keeping inside [0,1]
    func expandNormalizedBox(_ box: CGRect, by percent_w: CGFloat, by percent_h: CGFloat) -> CGRect {
     //   print("percent_w=\(percent_w), percent_h=\(percent_h)")
        guard percent_w > 0 || percent_h > 0 else { return box }
            let w = box.width
            let h = box.height
            let dw = w * percent_w/100
            let dh = h * percent_h/100
            let nx = box.origin.x + dw / 2.0
            let ny = box.origin.y + dh / 2.0
            let nw = w - dw
            let nh = h - dh
            return CGRect(x: nx, y: ny, width: nw, height: nh)
    }
    
    
    let minimumZoom: CGFloat = 1.0
    let maximumZoom: CGFloat = 10.0
    var lastZoomFactor: CGFloat = 1.0

    @IBAction func pinch(_ pinch: UIPinchGestureRecognizer) {
      let device = videoCapture.captureDevice

      // Return zoom value between the minimum and maximum zoom values
      func minMaxZoom(_ factor: CGFloat) -> CGFloat {
        return min(min(max(factor, minimumZoom), maximumZoom), device.activeFormat.videoMaxZoomFactor)
      }

      func update(scale factor: CGFloat) {
        do {
          try device.lockForConfiguration()
          defer {
            device.unlockForConfiguration()
          }
          device.videoZoomFactor = factor
        } catch {
          print("\(error.localizedDescription)")
        }
      }

      let newScaleFactor = minMaxZoom(pinch.scale * lastZoomFactor)
      switch pinch.state {
      case .began, .changed:
        update(scale: newScaleFactor)
        self.labelZoom.text = String(format: "%.2fx", newScaleFactor)
        self.labelZoom.font = UIFont.preferredFont(forTextStyle: .title2)
      case .ended:
        lastZoomFactor = minMaxZoom(newScaleFactor)
        update(scale: lastZoomFactor)
        self.labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
      default: break
      }
    }  // Pinch to Zoom End
    
    @objc private func boundingBoxTapped(_ sender: UITapGestureRecognizer) {
        guard let tappedView = sender.view as? BoundingBoxView else { return }
        guard boundingBoxViews.firstIndex(of: tappedView) != nil else { return }
        // Получаем текст номера из BoundingBoxView (label)
        let plateText = tappedView.labelText ?? ""
        guard !plateText.isEmpty else { return }
        let vc = DetectedPlateViewController(plate: plateText)
        vc.modalPresentationStyle = .formSheet
        present(vc, animated: true, completion: nil)
    }
 }

// Extension placed outside of class
extension CVPixelBuffer {
    func toUIImage(orientation: UIImage.Orientation = .up) -> UIImage? {
        let ciImage = CIImage(cvPixelBuffer: self)
        guard let cgImage = sharedCIContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }
        return UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation)
    }
}

// Re-open class to add remaining helper methods
extension ViewController {
    
    //
    func normalizedBoundingBoxForUIImageAI(boundingBox: CGRect) -> CGRect {

        guard let previewLayer = videoCapture.previewLayer else {
            return .zero
        }

        return previewLayer.layerRectConverted(
            fromMetadataOutputRect: boundingBox
        )
    }
    
    // Map a Vision boundingBox (normalized) to normalized coordinates relative to the uiImage
    // taking into account how the image is displayed inside videoPreview (letterbox/pillarbox)
    func normalizedBoundingBoxForUIImage(boundingBox: CGRect, videoPreview: UIView) -> CGRect {
        
        let width = videoPreview.bounds.width  // 375 pix
        let height = videoPreview.bounds.height  // 812 pix
        // ratio = videoPreview AR divided by sessionPreset AR
        var ratio: CGFloat = 1.0
        if videoCapture.captureSession.sessionPreset == .photo {
          ratio = (height / width) / (4.0 / 3.0)  // .photo
        } else {
          ratio = (height / width) / (16.0 / 9.0)  // .hd4K3840x2160, .hd1920x1080, .hd1280x720 etc.
        }
        
        var rect = boundingBox
 //       if developerMode {
 //           print("1 Original box: \(rect)")
 //       }
        switch UIDevice.current.orientation {
        case .portraitUpsideDown:
          rect = CGRect(
            x: 1.0 - rect.origin.x - rect.width,
            y: 1.0 - rect.origin.y - rect.height,
            width: rect.width,
            height: rect.height)
        case .landscapeLeft:
          rect = CGRect(
            x: rect.origin.x,
            y: rect.origin.y,
            width: rect.width,
            height: rect.height)
        case .landscapeRight:
          rect = CGRect(
            x: rect.origin.x,
            y: rect.origin.y,
            width: rect.width,
            height: rect.height)
        case .unknown:
          print("The device orientation is unknown, the predictions may be affected")
          fallthrough
        default: break
        }
        
 //       if developerMode {
 //           print("2 Adjusted for orientation: \(rect)")
 //       }
            if ratio >= 1 {  // iPhone ratio = 1.218
          let offset = (1 - ratio) * (0.5 - rect.minX)
          let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
          rect = rect.applying(transform)
          rect.size.width *= ratio
        } else {  // iPad ratio = 0.75
          let offset = (ratio - 1) * (0.5 - rect.maxY)
          let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
          rect = rect.applying(transform)
          ratio = (height / width) / (3.0 / 4.0)
          rect.size.height /= ratio
        }
 //       if developerMode {
 //           print("3 Adjusted for aspect ratio: \(rect)")
 //       }
        // Scale normalized to pixels [375, 812] [width, height]
        rect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))
 //       if developerMode {
 //           print("4 Scaled to pixels: \(rect)")
 //       }
        return rect
    }

    // Adjust normalized box for UIImage orientation
    func adjustNormBoxForImageOrientation(normBox: CGRect, orientation: UIImage.Orientation) -> CGRect {
        switch orientation {
        case .right:
            // 90° CW
            return CGRect(
                x: 1.0 - normBox.origin.y - normBox.size.height,
                y: normBox.origin.x,
                width: normBox.size.height,
                height: normBox.size.width
            )
        case .left:
            // 270° CW
            return CGRect(
                x: normBox.origin.y,
                y: 1.0 - normBox.origin.x - normBox.size.width,
                width: normBox.size.height,
                height: normBox.size.width
            )
        case .down:
            // 180°
            return CGRect(
                x: 1.0 - normBox.origin.x - normBox.size.width,
                y: 1.0 - normBox.origin.y - normBox.size.height,
                width: normBox.size.width,
                height: normBox.size.height
            )
        default:
            return normBox
        }
    }
    
    //----
    func normalizedBoundingBoxForUIImage2(boundingBox: CGRect, uiImage: UIImage) -> CGRect {

        let imageWidth = uiImage.size.width
        let imageHeight = uiImage.size.height

        // Vision box → UIKit coords
        let rect = CGRect(
            x: boundingBox.origin.x * imageWidth,
            y: (1 - boundingBox.origin.y - boundingBox.height) * imageHeight,
            width: boundingBox.width * imageWidth,
            height: boundingBox.height * imageHeight
        )

        return rect
    }

    // Compute the rect in the view where the image is actually displayed (accounting for letterbox/pillarbox)
    func imageDisplayRectInView(imageSize: CGSize, viewSize: CGSize) -> CGRect {
        let imageAspect = imageSize.width / imageSize.height
        let viewAspect = viewSize.width / viewSize.height
        if imageAspect > viewAspect {
            // fit by width -> vertical letterboxing
            let scale = viewSize.width / imageSize.width
            let displayHeight = imageSize.height * scale
            let y = (viewSize.height - displayHeight) / 2.0
            return CGRect(x: 0, y: y, width: viewSize.width, height: displayHeight)
        } else {
            // fit by height -> horizontal pillarboxing
            let scale = viewSize.height / imageSize.height
            let displayWidth = imageSize.width * scale
            let x = (viewSize.width - displayWidth) / 2.0
            return CGRect(x: x, y: 0, width: displayWidth, height: viewSize.height)
        }
    }

   
    // Preprocess image for OCR: convert to grayscale, increase contrast and exposure
    func preprocessImageForOCR(_ image: UIImage) -> UIImage {
        guard let ciImage = CIImage(image: image) else { return image }
        // Convert to grayscale
        let gray = ciImage.applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0.0])
        // Increase contrast and exposure slightly
        let contrasted = gray.applyingFilter("CIColorControls", parameters: [kCIInputContrastKey: 1.2])
        let exposed = contrasted.applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: 0.2])
        // Optionally apply a small sharpening
        let sharpened = exposed.applyingFilter("CISharpenLuminance", parameters: [kCIInputSharpnessKey: 0.4])
        if let cg = sharedCIContext.createCGImage(sharpened, from: sharpened.extent) {
            return UIImage(cgImage: cg, scale: image.scale, orientation: image.imageOrientation)
        }
        return image
    }
    //---------
    func preprocessPlateForOCR2(_ image: UIImage) -> UIImage {
        // Start with a CIImage for filtering
        guard let inputCI = CIImage(image: image) else { return image }

        // 1) Grayscale
        let gray = inputCI.applyingFilter("CIColorControls", parameters: [kCIInputSaturationKey: 0.0])
        // 2) Increase contrast
        let contrasted = gray.applyingFilter("CIColorControls", parameters: [kCIInputContrastKey: 1.5])
        // 3) Slight exposure
        let exposed = contrasted.applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: 0.2])
        // 4) Slight sharpening
        let sharpened = exposed.applyingFilter("CISharpenLuminance", parameters: [kCIInputSharpnessKey: 0.5])

        // 5) Simple threshold via color clamp
        /*   let thresholdFilter = CIFilter(name: "CIColorClamp")!
        thresholdFilter.setValue(sharpened, forKey: kCIInputImageKey)
        thresholdFilter.setValue(CIVector(x: 0.3, y: 0.3, z: 0.3, w: 0), forKey: "inputMinComponents")
        thresholdFilter.setValue(CIVector(x: 1, y: 1, z: 1, w: 1), forKey: "inputMaxComponents")
        let thresholded = thresholdFilter.outputImage ?? sharpened
        */
        // Вместо CIColorClamp:
        let thresholded = sharpened
            .applyingFilter("CIUnsharpMask", parameters: [
                kCIInputRadiusKey: 1.2,
                kCIInputIntensityKey: 0.5
            ])
            .applyingFilter("CIHighlightShadowAdjust", parameters: [
                "inputShadowAmount": 0.2, // приподнять тени
                "inputHighlightAmount": 0.0
            ])

        // 6) Create UIImage from CIImage
        guard let cgOut = sharedCIContext.createCGImage(thresholded, from: thresholded.extent) else {
            return image
        }
        let processedUIImage = UIImage(cgImage: cgOut, scale: image.scale, orientation: image.imageOrientation)

        // 7) Scale to a reasonable height for OCR (>= 60px)
        let targetHeight: CGFloat = 60
        let currentHeight = processedUIImage.size.height
        if currentHeight >= targetHeight {
            return processedUIImage
        }
        let scale = targetHeight / max(currentHeight, 1)
        let newSize = CGSize(width: processedUIImage.size.width * scale, height: targetHeight)
        UIGraphicsBeginImageContextWithOptions(newSize, false, 1.0)
        processedUIImage.draw(in: CGRect(origin: .zero, size: newSize))
        let scaledImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return scaledImage ?? processedUIImage
    }
}

extension ViewController {

    func photoOutput(
      _ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
    ) {
      if let error = error {
        print("error occurred : \(error.localizedDescription)")
      }
      if let dataImage = photo.fileDataRepresentation() {
        let dataProvider = CGDataProvider(data: dataImage as CFData)
        let cgImageRef: CGImage! = CGImage(
          jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true,
          intent: .defaultIntent)
        var orientation = CGImagePropertyOrientation.right
        switch UIDevice.current.orientation {
        case .landscapeLeft:
          orientation = .up
        case .landscapeRight:
          orientation = .down
        default:
          break
        }
        var image = UIImage(cgImage: cgImageRef, scale: 0.5, orientation: .right)
        if let orientedCIImage = CIImage(image: image)?.oriented(orientation),
          let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent)
        {
          image = UIImage(cgImage: cgImage)
        }
        let imageView = UIImageView(image: image)
        imageView.contentMode = .scaleAspectFill
        imageView.frame = videoPreview.frame
        let imageLayer = imageView.layer
        videoPreview.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)

        let bounds = UIScreen.main.bounds
        UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
        self.View0.drawHierarchy(in: bounds, afterScreenUpdates: true)
        let img = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        imageLayer.removeFromSuperlayer()
        let activityViewController = UIActivityViewController(
          activityItems: [img!], applicationActivities: nil)
        activityViewController.popoverPresentationController?.sourceView = self.View0
        self.present(activityViewController, animated: true, completion: nil)
        //
        //            // Save to camera roll
        //            UIImageWriteToSavedPhotosAlbum(img!, nil, nil, nil);
      } else {
        print("AVCapturePhotoCaptureDelegate Error")
      }
    }

}
