
import UIKit

protocol DetectedPlateViewControllerDelegate: AnyObject {
    func detectedPlateDidClose()
}

class DetectedPlateViewController: UIViewController {

    weak var delegate: DetectedPlateViewControllerDelegate?

    var plateText: String = ""

    // MARK: - UI

    private let titleLabel: UILabel = {
        let l = UILabel()
        l.translatesAutoresizingMaskIntoConstraints = false
        l.font = UIFont.systemFont(ofSize: 20, weight: .semibold)
        l.textAlignment = .center
        l.text = "Detected plate"
        return l
    }()

    private let plateLabel: UILabel = {
        let l = UILabel()
        l.translatesAutoresizingMaskIntoConstraints = false
        l.font = UIFont.monospacedDigitSystemFont(ofSize: 36, weight: .bold)
        l.textAlignment = .center
        l.numberOfLines = 1
        l.adjustsFontSizeToFitWidth = true
        return l
    }()

    private let infoLabel: UILabel = {
        let l = UILabel()
        l.translatesAutoresizingMaskIntoConstraints = false
        l.font = UIFont.systemFont(ofSize: 16, weight: .regular)
        l.textAlignment = .center
        l.textColor = .secondaryLabel
        l.numberOfLines = 0
        l.text = "Here you will be able to process the recognized number!"
        return l
    }()

    private let closeButton: UIButton = {
        let b = UIButton(type: .system)
        b.translatesAutoresizingMaskIntoConstraints = false
        b.setTitle("Close", for: .normal)
        b.titleLabel?.font = UIFont.systemFont(ofSize: 17, weight: .medium)
        return b
    }()

    // MARK: - Init

    init(plate: String) {
        self.plateText = plate
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    // MARK: - Lifecycle

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .systemBackground
        setupLayout()
        plateLabel.text = plateText
        closeButton.addTarget(self, action: #selector(closeTapped), for: .touchUpInside)
    }

    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        delegate?.detectedPlateDidClose()
    }

    // MARK: - Layout

    private func setupLayout() {
        view.addSubview(titleLabel)
        view.addSubview(plateLabel)
        view.addSubview(infoLabel)
        view.addSubview(closeButton)

        NSLayoutConstraint.activate([
            titleLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 32),
            titleLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            titleLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24),

            plateLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 24),
            plateLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            plateLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24),

            infoLabel.topAnchor.constraint(equalTo: plateLabel.bottomAnchor, constant: 20),
            infoLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 24),
            infoLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -24),

            closeButton.topAnchor.constraint(equalTo: infoLabel.bottomAnchor, constant: 32),
            closeButton.centerXAnchor.constraint(equalTo: view.centerXAnchor)
        ])
    }

    // MARK: - Actions

    @objc private func closeTapped() {
        dismiss(animated: true)
    }
}
