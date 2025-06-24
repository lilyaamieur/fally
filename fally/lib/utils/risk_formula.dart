import 'dart:math';

enum RiskLevel { critical, high, moderate, low }
enum AttachmentForm { uShaped, vShaped }

class BranchAssessmentResult {
  final RiskLevel riskLevel;
  final double diameterRatio;
  final double predictedStress;
  final bool requiresAction;
  final List<String> actionRecommendations; // Changed from String to List<String>
  final double? minimumRemovalLength;
  final double? optimalRemovalLength;
  final bool emergencyClosure;

  BranchAssessmentResult({
    required this.riskLevel,
    required this.diameterRatio,
    required this.predictedStress,
    required this.requiresAction,
    required this.actionRecommendations, // Changed parameter
    this.minimumRemovalLength,
    this.optimalRemovalLength,
    required this.emergencyClosure,
  });

  @override
  String toString() {
    String result = 'Risk Level: ${riskLevel.name.toUpperCase()}\n';
    result += 'Diameter Ratio: ${diameterRatio.toStringAsFixed(3)}\n';
    result += 'Predicted Stress: ${predictedStress.toStringAsFixed(1)} MPa\n';
    result += 'Emergency Closure Required: ${emergencyClosure ? "YES" : "NO"}\n';
    result += 'Actions:\n';
    for (String action in actionRecommendations) {
      result += '  • $action\n';
    }

    if (minimumRemovalLength != null) {
      result += 'Minimum Removal Length: ${minimumRemovalLength!.toStringAsFixed(2)}m\n';
    }
    if (optimalRemovalLength != null) {
      result += 'Optimal Removal Length: ${optimalRemovalLength!.toStringAsFixed(2)}m\n';
    }

    return result;
  }
}

/// Comprehensive branch risk assessment function based on research protocol
/// for century-old Ficus retusa trees
BranchAssessmentResult assessBranchRisk({
  required double branchDiameter, // cm, inside bark diameter
  required double trunkDiameterAbove, // cm, 30cm above attachment
  required double trunkDiameterBelow, // cm, 30cm below attachment
  required double branchLength, // meters, from attachment to tip
  required double branchAngleFromHorizontal, // degrees
  required AttachmentForm attachmentForm,
  required bool hasIncludedBark,
}) {
  // Step 1: Calculate average trunk diameter
  double avgTrunkDiameter = (trunkDiameterAbove + trunkDiameterBelow) / 2;

  // Step 2: Calculate diameter ratio (primary risk indicator)
  double diameterRatio = branchDiameter / avgTrunkDiameter;

  // Step 3: Apply risk multiplier for attachment form
  double riskMultiplier = 1.0;
  if (attachmentForm == AttachmentForm.vShaped) {
    riskMultiplier = 1.3;
  }
  if (hasIncludedBark) {
    riskMultiplier *= 1.2; // Additional risk for included bark
  }

  double adjustedDR = diameterRatio * riskMultiplier;

  // Step 4: Determine risk level based on research thresholds
  RiskLevel riskLevel;
  if (adjustedDR >= 0.70) {
    riskLevel = RiskLevel.critical;
  } else if (adjustedDR >= 0.50) {
    riskLevel = RiskLevel.high;
  } else if (adjustedDR >= 0.30) {
    riskLevel = RiskLevel.moderate;
  } else {
    riskLevel = RiskLevel.low;
  }

  // Step 5: Calculate predicted breaking stress (conservative approach)
  // Using: σ = 95 - 76 × DR (most conservative equation from research)
  double predictedStress = 95 - (76 * diameterRatio);

  // Step 6: Check for emergency closure conditions
  bool emergencyClosure = (diameterRatio >= 0.90) ||
      (diameterRatio >= 0.70 && branchAngleFromHorizontal < 30);

  // Step 7: Calculate pruning recommendations for unsafe branches
  double? minimumRemovalLength;
  double? optimalRemovalLength;

  if (riskLevel == RiskLevel.critical || riskLevel == RiskLevel.high) {
    // More conservative approach that preserves aesthetics
    // Target: Reduce effective diameter ratio to 0.60 (safer but less aggressive)
    double targetDR = 0.60;

    // For tapering branches, we estimate diameter reduction with distance
    // Using more conservative taper rate for mature Ficus trees
    double taperRate = 0.015; // Slower taper for old growth

    // Calculate length needed to reduce diameter to target
    double targetDiameter = targetDR * avgTrunkDiameter / riskMultiplier;

    if (targetDiameter < branchDiameter && branchLength > 2.0) {
      double diameterReduction = branchDiameter - targetDiameter;
      double calculatedLength = diameterReduction / (branchDiameter * taperRate);

      // Much more conservative limits to preserve aesthetics
      // Never remove more than 25% of branch length
      minimumRemovalLength = min(calculatedLength, branchLength * 0.25);

      // Optimal removal is only slightly more (30% max)
      optimalRemovalLength = min(minimumRemovalLength * 1.1, branchLength * 0.30);

      // Absolute minimum cut - at least 1m for meaningful risk reduction
      if (minimumRemovalLength != null && minimumRemovalLength! < 1.0) {
        minimumRemovalLength = min(1.0, branchLength * 0.20);
        optimalRemovalLength = min(1.2, branchLength * 0.25);
      }
    } else {
      // For shorter branches or uniform thickness
      // Very conservative - only remove 15-20% maximum
      minimumRemovalLength = branchLength * 0.15;
      optimalRemovalLength = branchLength * 0.20;
    }

    // Additional aesthetic constraint: never remove more than 3 meters
    // (preserve major branch structure)
    if (minimumRemovalLength != null && minimumRemovalLength! > 3.0) {
      minimumRemovalLength = 3.0;
      optimalRemovalLength = 3.5;
    }
  }

  // Step 8: Generate action recommendations as a list
  List<String> actionRecommendations = [];
  bool requiresAction = true;

  switch (riskLevel) {
    case RiskLevel.critical:
      if (emergencyClosure) {
        actionRecommendations.add("EMERGENCY: Close area immediately");
        actionRecommendations.add("Remove branch or reduce by ${minimumRemovalLength?.toStringAsFixed(1)}m");
      } else {
        actionRecommendations.add("CRITICAL: Light reduction of ${minimumRemovalLength?.toStringAsFixed(1)}m within 7 days");
        actionRecommendations.add("Action preserves branch structure");
      }
      break;
    case RiskLevel.high:
      actionRecommendations.add("HIGH: Conservative pruning of ${minimumRemovalLength?.toStringAsFixed(1)}m within 30 days");
      actionRecommendations.add("Maintains tree aesthetics");
      break;
    case RiskLevel.moderate:
      actionRecommendations.add("MODERATE: Monitor quarterly");
      actionRecommendations.add("Consider light pruning for prevention");
      requiresAction = false;
      break;
    case RiskLevel.low:
      actionRecommendations.add("LOW: Annual inspection sufficient");
      requiresAction = false;
      break;
  }

  // Add specific guidance for attachment issues
  if (hasIncludedBark && (riskLevel == RiskLevel.critical || riskLevel == RiskLevel.high)) {
    actionRecommendations.add("Included bark increases failure risk");
  }

  if (attachmentForm == AttachmentForm.vShaped && (riskLevel == RiskLevel.critical || riskLevel == RiskLevel.high)) {
    actionRecommendations.add("V-shaped attachment requires priority attention");
  }

  return BranchAssessmentResult(
    riskLevel: riskLevel,
    diameterRatio: diameterRatio,
    predictedStress: predictedStress,
    requiresAction: requiresAction,
    actionRecommendations: actionRecommendations,
    minimumRemovalLength: minimumRemovalLength,
    optimalRemovalLength: optimalRemovalLength,
    emergencyClosure: emergencyClosure,
  );
}

/// Additional utility function to calculate breaking stress under load
double calculateBreakingStress({
  required double appliedLoad, // kN (use 1.0 for baseline assessment)
  required double leverArm, // meters, distance from load to attachment
  required double branchAngle, // degrees from horizontal
  required double branchDiameter, // meters, inside bark
}) {
  // Convert angle to radians
  double angleRad = branchAngle * pi / 180;

  // Calculate breaking stress using research formula
  // σ = (P × L × cos(θ)) / (π × d³ / 32)
  double stress = (appliedLoad * leverArm * cos(angleRad)) /
      (pi * pow(branchDiameter, 3) / 32);

  return stress; // Returns stress in MPa (assuming load in kN, distance in m)
}

/// Main function - entry point for the program
void main() {
  demonstrateAssessment();
}

/// Example usage and testing function
void demonstrateAssessment() {
  print("=== Ficus Branch Risk Assessment Demo ===\n");

  // Example 1: Critical risk branch
  var result1 = assessBranchRisk(
    branchDiameter: 35.0, // 35cm diameter
    trunkDiameterAbove: 45.0,
    trunkDiameterBelow: 47.0,
    branchLength: 8.0, // 8 meters long
    branchAngleFromHorizontal: 25.0, // Low angle - more dangerous
    attachmentForm: AttachmentForm.vShaped,
    hasIncludedBark: true,
  );

  print("Example 1 - Large V-shaped branch with included bark:");
  print(result1);
  print("");

  // Example 2: Moderate risk branch
  var result2 = assessBranchRisk(
    branchDiameter: 18.0, // 18cm diameter
    trunkDiameterAbove: 52.0,
    trunkDiameterBelow: 48.0,
    branchLength: 6.0,
    branchAngleFromHorizontal: 45.0,
    attachmentForm: AttachmentForm.uShaped,
    hasIncludedBark: false,
  );

  print("Example 2 - Medium U-shaped branch without included bark:");
  print(result2);
  print("");

  // Example 3: Safe branch
  var result3 = assessBranchRisk(
    branchDiameter: 12.0, // 12cm diameter
    trunkDiameterAbove: 55.0,
    trunkDiameterBelow: 53.0,
    branchLength: 4.0,
    branchAngleFromHorizontal: 60.0,
    attachmentForm: AttachmentForm.uShaped,
    hasIncludedBark: false,
  );

  print("Example 3 - Small well-attached branch:");
  print(result3);
  print("");

  // Demonstrate stress calculation
  print("=== Stress Calculation Example ===");
  double stress = calculateBreakingStress(
    appliedLoad: 1.0, // 1 kN baseline load
    leverArm: 3.0, // 3 meters from attachment
    branchAngle: 30.0, // 30 degrees from horizontal
    branchDiameter: 0.25, // 25cm = 0.25m
  );
  print("Breaking stress for 25cm branch with 1kN load at 3m: ${stress.toStringAsFixed(2)} MPa");
}