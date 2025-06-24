import 'package:fally/style/theme.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'dart:math' as math;

//import 'risk_as.dart';
import '../../utils/risk_formula.dart';

class BranchAssessmentPage extends StatefulWidget {
  @override
  _BranchAssessmentPageState createState() => _BranchAssessmentPageState();
}

class _BranchAssessmentPageState extends State<BranchAssessmentPage> {
  // Controllers for input fields
  final TextEditingController _treeIdController = TextEditingController();
  final TextEditingController _gpsController = TextEditingController();
  final TextEditingController _branchDiameterController =
      TextEditingController();
  final TextEditingController _trunkDiameterAboveController =
      TextEditingController();
  final TextEditingController _trunkDiameterBelowController =
      TextEditingController();
  final TextEditingController _branchAngleController = TextEditingController();
  final TextEditingController _branchLengthController = TextEditingController();
  final TextEditingController _appliedLoadController = TextEditingController();
  final TextEditingController _distanceToLoadController =
      TextEditingController();

  // State variables
  bool _expertMode = false;
  String _attachmentForm = 'U-Shaped';
  bool _includedBark = false;
  bool _showResults = false;
  File? _selectedImage;

  // Calculated values
  double _averageTrunkDiameter = 0.0;
  double _diameterRatio = 0.0;
  double _estimatedBreakingStress = 0.0;
  double _appliedStress = 0.0;
  String _riskClassification = '';
  String _recommendedAction = '';
  String _safetyDecision = '';
  Color _riskColor = Colors.grey;

  // Assessment results
  bool _requiresAction = false;
  RiskLevel? _riskLevel;
  List<String> _actionRecommendations = [];
  double? _minRemovalLength;
  double? _optRemovalLength;
  bool _emergencyClosure = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: const Text(
          'Branch Risk Assessment',
          style: TextStyle(color: Color.fromARGB(255, 102, 151, 102)),
        ),
        backgroundColor: Colors.white,
        elevation: 1,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Progress indicator -------------> to be done
              // Center(
              //   child: Container(
              //     width: 300,
              //     height: 25,
              //     decoration: BoxDecoration(
              //       color: const Color(0xFF8FBC8F),
              //       borderRadius: BorderRadius.circular(12),
              //       border: Border.all(color: const Color(0xFF8FBC8F), width: 2),
              //     ),
              //     child: Row(
              //       mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              //       children: [
              //         _circleBox(Colors.white),
              //         _circleBox(Colors.red),
              //         _circleBox(Colors.red),
              //       ],
              //     ),
              //   ),
              // ),
              const SizedBox(height: 20),

              // Tree Identification Section
              _buildSectionTitle("Tree Identification"),
              const SizedBox(height: 10),
              _buildTreeIdentificationSection(),

              const SizedBox(height: 30),

              // Physical Measurements Section
              _buildSectionTitle("Branch Measurements"),
              const SizedBox(height: 10),
              _buildMeasurementsSection(),

              const SizedBox(height: 20),

              // Structural Attachment Information (Expert Mode)
              if (true) ...[
                // _buildSectionTitle("Structural Attachment Information"),
                const SizedBox(height: 10),
                // _buildStructuralAttachmentSection(),
                const SizedBox(height: 20),
              ],

              // Action Buttons
              _buildActionButtons(),

              const SizedBox(height: 20),

              // Results Section
              if (_showResults) _buildResultsSection(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSectionTitle(String title) {
    return Text(
      title,
      style: const TextStyle(
        fontWeight: FontWeight.bold,
        fontSize: 18,
        color: AppColors.darkGreen,
      ),
    );
  }

  Widget _buildTreeIdentificationSection() {
    return Center(
      child: Container(
        height: 190,
        decoration: BoxDecoration(
          color: AppColors.creamyWhite,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.lightGreen, width: 2),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Left Side: Inputs
            Expanded(
              child: Padding(
                padding: const EdgeInsets.fromLTRB(12, 30, 6, 16),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Row(
                      children: [
                        const Text(
                          'Tree ID:',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        const SizedBox(width: 10),
                        // spacing between label and container
                        Expanded(
                          // ensures the input takes remaining space
                          child: Container(
                            decoration: BoxDecoration(
                              color: const Color.fromARGB(0, 83, 131, 83),
                              borderRadius: BorderRadius.circular(17),
                              border: Border.all(
                                color: Colors.grey, // your grey border
                                width: 1.5, // optional: adjust thickness
                              ),
                            ),
                            padding: const EdgeInsets.symmetric(horizontal: 12),
                            // optional padding
                            child: TextFormField(
                              controller: _treeIdController,
                              decoration: const InputDecoration(
                                border: InputBorder.none, // removes underline
                                labelStyle: TextStyle(
                                  color: Colors.black,
                                  fontSize: 30,
                                ),
                                hintText: 'enter the ID',
                              ),
                              style: const TextStyle(color: Colors.black),
                            ),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    Row(
                      children: [
                        const Icon(Icons.location_on, color: Colors.black),
                        const SizedBox(width: 10),
                        // spacing between label and container
                        Expanded(
                          // ensures the input takes remaining space
                          child: Container(
                            decoration: BoxDecoration(
                              color: const Color.fromARGB(0, 83, 131, 83),
                              borderRadius: BorderRadius.circular(17),
                              border: Border.all(
                                color: Colors.grey, // your grey border
                                width: 1.5, // optional: adjust thickness
                              ),
                            ),
                            padding: const EdgeInsets.symmetric(horizontal: 12),
                            // optional padding
                            child: TextFormField(
                              controller: _gpsController,
                              decoration: const InputDecoration(
                                border: InputBorder.none, // removes underline
                                labelStyle: TextStyle(
                                  color: Colors.white,
                                  fontSize: 30,
                                ),
                                hintText: "Location",
                              ),
                              style: const TextStyle(color: Colors.black),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            // Right Side: Photo Upload Container
            Container(
              height: 160,
              width: 140,
              margin: const EdgeInsets.all(12.0),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Colors.grey.shade300, width: 2),
              ),
              child: InkWell(
                onTap: _uploadPhoto,
                child: _selectedImage == null
                    ? const Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.camera_alt, size: 40, color: Colors.grey),
                          SizedBox(height: 8),
                          Text(
                            'Upload Tree Photo',
                            style: TextStyle(color: Colors.grey, fontSize: 12),
                            textAlign: TextAlign.center,
                          ),
                        ],
                      )
                    : ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: Image.file(
                          _selectedImage!,
                          fit: BoxFit.cover,
                          width: double.infinity,
                          height: double.infinity,
                        ),
                      ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  //-------------------------------------------------//
  Widget _buildMeasurementsSection() {
    return Center(
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: AppColors.creamyWhite,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppColors.lightGreen, width: 2),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              margin: const EdgeInsets.fromLTRB(0, 0, 0, 10),
              child: const Text(
                "Branch identification:",
                style: TextStyle(
                  color: Color.fromRGBO(67, 182, 61, 1),
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            Row(
              children: [
                const Text(
                  'Branch ID:',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    child: Container(
                      decoration: BoxDecoration(
                        color: const Color.fromARGB(0, 83, 131, 83),
                        borderRadius: BorderRadius.circular(17),
                        border: Border.all(
                          color: Colors.grey, // your grey border
                          width: 1.5, // optional: adjust thickness
                        ),
                      ),
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      // Inner padding
                      child: TextFormField(
                        controller: _branchDiameterController,
                        keyboardType: TextInputType.number,
                        decoration: const InputDecoration(
                          labelStyle: TextStyle(
                            color: Colors.white,
                            fontSize: 100,
                          ),
                          //--------> color is not being changed idk why
                          hintText: "diameter (cm)",
                          border: InputBorder.none,
                        ),
                        onChanged: (_) => _calculateValues(),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 20)
              ],
            ),
            const SizedBox(height: 20),
            Container(
              margin: const EdgeInsets.fromLTRB(0, 0, 0, 10),
              child: const Text(
                "Physical measurements:",
                style: TextStyle(
                  color: Color.fromRGBO(67, 182, 61, 1),
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
            // Branch Diameter
            Row(
              children: [
                const Text(
                  'Branch Diameter:',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    child: Container(
                      decoration: BoxDecoration(
                        color: const Color.fromARGB(0, 83, 131, 83),
                        borderRadius: BorderRadius.circular(17),
                        border: Border.all(
                          color: Colors.grey, // your grey border
                          width: 1.5, // optional: adjust thickness
                        ),
                      ),
                      padding: const EdgeInsets.symmetric(horizontal: 12),
                      // Inner padding
                      child: TextFormField(
                        controller: _branchDiameterController,
                        keyboardType: TextInputType.number,
                        decoration: const InputDecoration(
                          labelStyle: TextStyle(
                            color: Colors.white,
                            fontSize: 100,
                          ),
                          //--------> color is not being changed idk why
                          hintText: "diameter (cm)",
                          border: InputBorder.none,
                        ),
                        onChanged: (_) => _calculateValues(),
                      ),
                    ),
                  ),
                ),
              ],
            ),

            // Trunk Diameter Above
            Row(
              children: [
                const Text(
                  'Trunk Diameter Above:',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    child: Container(
                      decoration: BoxDecoration(
                        color: const Color.fromARGB(0, 83, 131, 83),
                        borderRadius: BorderRadius.circular(17),
                        border: Border.all(
                          color: Colors.grey, // your grey border
                          width: 1.5, // optional: adjust thickness
                        ),
                      ),
                      child: TextFormField(
                        controller: _trunkDiameterAboveController,
                        keyboardType: TextInputType.number,

                        decoration: const InputDecoration(
                          labelStyle: TextStyle(
                            color: Colors.white,
                            fontSize: 100,
                          ),
                          //--------> color is not being changed idk why
                          hintText: "diameter (cm)",
                          border: InputBorder.none,
                        ),
                        onChanged: (_) => _calculateValues(),
                      ),
                    ),
                  ),
                ),
              ],
            ),

            // Trunk Diameter Below
            Row(
              children: [
                const Text(
                  'Trunk Diameter Below:',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(vertical: 8),
                    child: Container(
                      decoration: BoxDecoration(
                        color: const Color.fromARGB(0, 83, 131, 83),
                        borderRadius: BorderRadius.circular(17),
                        border: Border.all(
                          color: Colors.grey, // your grey border
                          width: 1.5, // optional: adjust thickness
                        ),
                      ),
                      child: TextFormField(
                        controller: _trunkDiameterBelowController,
                        keyboardType: TextInputType.number,
                        decoration: const InputDecoration(
                          labelStyle: TextStyle(
                            color: Colors.white,
                            fontSize: 100,
                          ),
                          //--------> color is not being changed idk why
                          hintText: "diameter (cm)",
                          border: InputBorder.none,
                        ),
                        onChanged: (_) => _calculateValues(),
                      ),
                    ),
                  ),
                ),
              ],
            ),

            Container(
              margin: const EdgeInsets.fromLTRB(0, 30, 0, 20),
              child: const Text(
                "Structural measurements:",
                style: TextStyle(
                  color: Color.fromRGBO(67, 182, 61, 1),
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),

            Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                // Attachment Form label
                const Text(
                  'Attachment Form:',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                const SizedBox(width: 10),

                // Dropdown
                Expanded(
                  child: Container(
                    width: 100,
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    margin: const EdgeInsets.only(right: 30),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey),
                    ),
                    child: DropdownButtonFormField<String>(
                      value: _attachmentForm,
                      isDense: true,
                      isExpanded: false,
                      // prevents stretching text to full width
                      decoration: const InputDecoration(
                        border: InputBorder.none,
                        contentPadding: EdgeInsets.symmetric(horizontal: 0),
                      ),
                      icon: const Padding(
                        // Reduce icon left padding
                        padding: EdgeInsets.only(left: 4),
                        child: Icon(Icons.arrow_drop_down),
                      ),
                      items: ['U-Shaped', 'V-Shaped'].map((String value) {
                        return DropdownMenuItem<String>(
                          value: value,
                          child: Text(value, style: TextStyle(fontSize: 14)),
                        );
                      }).toList(),
                      onChanged: (String? newValue) {
                        setState(() {
                          _attachmentForm = newValue!;
                        });
                      },
                    ),
                  ),
                ),
                // Included Bark toggle
                // GestureDetector(
                //   onTap: () {
                //     setState(() {
                //       _includedBark = !_includedBark;
                //     });
                //   },
                //   child: Container(
                //     padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                //     decoration: BoxDecoration(
                //       color: Colors.white,
                //       borderRadius: BorderRadius.circular(12),
                //       border: Border.all(color: Colors.grey),
                //     ),
                //     child: Row(
                //       children: [
                //         Container(
                //           width: 20,
                //           height: 20,
                //           decoration: BoxDecoration(
                //             shape: BoxShape.circle,
                //             color: _includedBark ? Colors.green : Colors.transparent,
                //             border: Border.all(color: Colors.grey),
                //           ),
                //         ),
                //         const SizedBox(width: 8),
                //         Text(
                //           'Included Bark',
                //           style: TextStyle(
                //             fontSize: 16,
                //             color: _includedBark ? Colors.green : Colors.black,
                //           ),
                //         ),
                //       ],
                //     ),
                //   ),
                // ),
              ],
            ),

            const SizedBox(height: 16),
            // Included Bark toggle
            Row(
              children: [
                const Text(
                  'Bark:',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                const SizedBox(width: 10),
                GestureDetector(
                  onTap: () {
                    setState(() {
                      _includedBark = !_includedBark;
                    });
                  },
                  child: Row(
                    children: [
                      Container(
                        width: 20,
                        height: 20,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _includedBark
                              ? Colors.green
                              : Colors.transparent,
                          border: Border.all(color: Colors.grey),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'Included',
                        style: TextStyle(
                          fontSize: 16,
                          color: _includedBark ? Colors.green : Colors.black,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            // Branch Angle
            Row(
              children: [
                const Text(
                  'Branch Angle:',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey),
                    ),
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    child: TextFormField(
                      controller: _branchAngleController,
                      keyboardType: TextInputType.number,
                      decoration: const InputDecoration(
                        hintText: 'Degrees',
                        border: InputBorder.none,
                      ),
                      onChanged: (_) => _calculateValues(),
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),

            // Branch Length
            Row(
              children: [
                const Text(
                  'Branch Length:',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey),
                    ),
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    child: TextFormField(
                      controller: _branchLengthController,
                      keyboardType: TextInputType.number,
                      decoration: const InputDecoration(
                        hintText: 'Length (m) - Optional',
                        border: InputBorder.none,
                      ),
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),

            /*// Applied Load
            Row(
              children: [
                const Text(
                  'Applied Load:',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey),
                    ),
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    child: TextFormField(
                      controller: _appliedLoadController,
                      keyboardType: TextInputType.number,
                      decoration: const InputDecoration(
                        hintText: 'kN',
                        border: InputBorder.none,
                      ),
                      onChanged: (_) => _calculateValues(),
                    ),
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),

            // Distance to Load
            Row(
              children: [
                const Text(
                  'Distance to Load:',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey),
                    ),
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    child: TextFormField(
                      controller: _distanceToLoadController,
                      keyboardType: TextInputType.number,
                      decoration: const InputDecoration(
                        hintText: 'Distance (m)',
                        border: InputBorder.none,
                      ),
                      onChanged: (_) => _calculateValues(),
                    ),
                  ),
                ),
              ],
            ),*/
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  Widget _buildActionButtons() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        ElevatedButton.icon(
          onPressed: _assessRisk,
          icon: const Icon(Icons.assessment, color: Colors.white),
          label: const Text(
            "Assess Risk",
            style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
          ),
          style: ElevatedButton.styleFrom(
            backgroundColor: AppColors.mediumGreen,
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            textStyle: const TextStyle(fontSize: 16),
          ),
        ),
        //_buildRiskResultWidget(),
        ElevatedButton.icon(
          onPressed: _saveToDatabase,
          icon: const Icon(Icons.save, color: Colors.white),
          label: const Text(
            "Save Assessment",
            style: TextStyle(color: Colors.white),
          ),
          style: ElevatedButton.styleFrom(
            backgroundColor: const Color(0x5671E071),
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
            textStyle: const TextStyle(fontSize: 16),
          ),
        ),
      ],
    );
  }

  Widget _buildResultRow(String label, String value, {Color? valueColor}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontWeight: FontWeight.w500)),
          Text(
            value,
            style: TextStyle(
              fontWeight: FontWeight.bold,
              color: valueColor ?? Colors.black87,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRiskResultWidget() {
    if (!_showResults || _riskLevel == null) return const SizedBox();

    return Container(
      margin: const EdgeInsets.only(top: 20),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border.all(
          color: _riskLevel == RiskLevel.critical
              ? Colors.red.shade700
              : _riskLevel == RiskLevel.high
              ? Colors.red
              : _riskLevel == RiskLevel.moderate
              ? Colors.orange
              : Colors.green,
          width: 2,
        ),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Risk Level: ${_riskLevel!.name.toUpperCase()}',
            style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          Text('Diameter Ratio: ${_diameterRatio.toStringAsFixed(3)}'),
          Text(
            'Predicted Stress: ${_estimatedBreakingStress.toStringAsFixed(1)} MPa',
          ),
          Text('Requires Action: ${_requiresAction ? "Yes" : "No"}'),

          // Actions as bullet points
          const Padding(
            padding: EdgeInsets.only(top: 8, bottom: 4),
            child: Text(
              'Actions:',
              style: TextStyle(fontWeight: FontWeight.w500),
            ),
          ),
          ..._actionRecommendations.map(
            (action) => Padding(
              padding: const EdgeInsets.only(left: 16, bottom: 2),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    '• ',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  Expanded(child: Text(action)),
                ],
              ),
            ),
          ),

          if (_minRemovalLength != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                'Minimum Removal Length: ${_minRemovalLength!.toStringAsFixed(2)} m',
              ),
            ),
          if (_optRemovalLength != null)
            Text(
              'Optimal Removal Length: ${_optRemovalLength!.toStringAsFixed(2)} m',
            ),
          if (_emergencyClosure)
            const Padding(
              padding: EdgeInsets.only(top: 8),
              child: Text(
                '⚠️ Emergency Closure Required!',
                style: TextStyle(
                  color: Colors.red,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildResultsSection() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _riskColor, width: 3),
        boxShadow: [
          BoxShadow(
            color: _riskColor.withOpacity(0.2),
            blurRadius: 8,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Center(
            child: Text(
              'Risk Assessment Results',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: AppColors.darkGreen,
              ),
            ),
          ),
          const SizedBox(height: 16),

          // Original measurements
          _buildResultRow(
            'Diameter Ratio (DR)',
            _diameterRatio.toStringAsFixed(3),
          ),
          _buildResultRow(
            'Estimated Breaking Stress',
            '${_estimatedBreakingStress.toStringAsFixed(2)} MPa',
          ),
          if (_expertMode)
            _buildResultRow(
              'Applied Stress',
              '${_appliedStress.toStringAsFixed(2)} MPa',
            ),

          // Additional calculated measures from the second function
          _buildResultRow(
            'Risk Level',
            _riskLevel!.name.toUpperCase(),
            valueColor: _riskLevel == RiskLevel.critical
                ? Colors.red.shade700
                : _riskLevel == RiskLevel.high
                ? Colors.red
                : _riskLevel == RiskLevel.moderate
                ? Colors.orange
                : Colors.green,
          ),
          _buildResultRow(
            'Requires Action',
            _requiresAction ? "Yes" : "No",
            valueColor: _requiresAction ? Colors.red : Colors.green,
          ),
          // Action Required as bullet points
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Action Required:',
                  style: TextStyle(fontWeight: FontWeight.w500),
                ),
                const SizedBox(height: 4),
                if (_actionRecommendations.isEmpty)
                  const Padding(
                    padding: EdgeInsets.only(left: 16, top: 2),
                    child: Text('No specific actions required'),
                  )
                else
                  ..._actionRecommendations.map(
                    (action) => Padding(
                      padding: const EdgeInsets.only(left: 16, top: 2),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            '• ',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          Expanded(
                            child: Text(
                              action,
                              style: const TextStyle(
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
              ],
            ),
          ),
          if (_minRemovalLength != null)
            _buildResultRow(
              'Minimum Removal Length',
              '${_minRemovalLength!.toStringAsFixed(2)} m',
            ),
          if (_optRemovalLength != null)
            _buildResultRow(
              'Optimal Removal Length',
              '${_optRemovalLength!.toStringAsFixed(2)} m',
            ),

          // Emergency closure warning with special styling
          if (_emergencyClosure)
            Container(
              margin: const EdgeInsets.only(top: 12),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.red.shade50,
                border: Border.all(color: Colors.red, width: 2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                children: [
                  Icon(Icons.warning, color: Colors.red, size: 24),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Emergency Closure Required!',
                      style: TextStyle(
                        color: Colors.red.shade700,
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _circleBox(Color color) {
    return Container(
      width: 15,
      height: 15,
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: const Color(0xFF8FBC8F), width: 2),
      ),
    );
  }

  void _calculateValues() {
    double branchDiameter =
        double.tryParse(_branchDiameterController.text) ?? 0.0;
    double trunkAbove =
        double.tryParse(_trunkDiameterAboveController.text) ?? 0.0;
    double trunkBelow =
        double.tryParse(_trunkDiameterBelowController.text) ?? 0.0;
    double branchLength = double.tryParse(_branchLengthController.text) ?? 0.0;
    double branchAngle = double.tryParse(_branchAngleController.text) ?? 0.0;

    final AttachmentForm form = _attachmentForm == 'V-Shaped'
        ? AttachmentForm.vShaped
        : AttachmentForm.uShaped;

    final result = assessBranchRisk(
      branchDiameter: branchDiameter,
      trunkDiameterAbove: trunkAbove,
      trunkDiameterBelow: trunkBelow,
      branchLength: branchLength,
      branchAngleFromHorizontal: branchAngle,
      attachmentForm: form,
      hasIncludedBark: _includedBark,
    );

    setState(() {
      _diameterRatio = result.diameterRatio;
      _estimatedBreakingStress = result.predictedStress;
      _requiresAction = result.requiresAction;
      _riskLevel = result.riskLevel;
      _actionRecommendations = result.actionRecommendations;
      _minRemovalLength = result.minimumRemovalLength;
      _optRemovalLength = result.optimalRemovalLength;
      _emergencyClosure = result.emergencyClosure;
    });
  }

  void _assessRisk() {
    _calculateValues();

    setState(() {
      _showResults = true; // Use this to toggle the result widget visibility
    });
  }

  void _getCurrentLocation() {
    // TODO: Implement GPS location fetching
    setState(() {
      _gpsController.text =
          "36.7538° N, 3.0588° E"; // Placeholder for Blida, Algeria
    });
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('GPS location fetched successfully')),
    );
  }

  Future<void> _uploadPhoto() async {
    final ImagePicker picker = ImagePicker();
    final XFile? pickedFile = await picker.pickImage(
      source: ImageSource.gallery,
    );

    if (pickedFile != null) {
      setState(() {
        _selectedImage = File(pickedFile.path);
      });

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Photo selected successfully')),
      );
    }
  }

  void _saveToDatabase() {
    // TODO: Implement database saving
    if (_treeIdController.text.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please enter Tree ID before saving')),
      );
      return;
    }

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Assessment saved to database successfully'),
        backgroundColor: AppColors.mediumGreen,
      ),
    );
  }

  @override
  void dispose() {
    _treeIdController.dispose();
    _gpsController.dispose();
    _branchDiameterController.dispose();
    _trunkDiameterAboveController.dispose();
    _trunkDiameterBelowController.dispose();
    _branchAngleController.dispose();
    _branchLengthController.dispose();
    _appliedLoadController.dispose();
    _distanceToLoadController.dispose();
    super.dispose();
  }
}
