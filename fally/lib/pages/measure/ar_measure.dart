import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:arkit_plugin/arkit_plugin.dart';
import 'package:vector_math/vector_math_64.dart' as vector;

class ArMeasur extends StatefulWidget {
  const ArMeasur({Key? key}) : super(key: key);

  @override
  _ArMeasurState createState() => _ArMeasurState();
}

class _ArMeasurState extends State<ArMeasur> {
  late ARKitController arkitController;
  ARKitPlane? plane;
  ARKitNode? node;
  String anchorId = '';

  @override
  void dispose() {
    arkitController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AR Measure'),
      ),
      body: ARKitSceneView(
        onARKitViewCreated: onARKitViewCreated,
        planeDetection: ARPlaneDetection.horizontal,
        enableTapRecognizer: false,
      ),
    );
  }

  void onARKitViewCreated(ARKitController controller) {
    arkitController = controller;
    arkitController.onAddNodeForAnchor = addAnchor;
    arkitController.onUpdateNodeForAnchor = updateAnchor;
  }

  void addAnchor(ARKitAnchor anchor) {
    if (anchor is! ARKitPlaneAnchor) return;
    addPlane(arkitController, anchor);
  }

  void addPlane(ARKitController controller, ARKitPlaneAnchor anchor) {
    anchorId = anchor.identifier;

    plane = ARKitPlane(
      width: anchor.extent.x,
      height: anchor.extent.z,
      materials: [
        ARKitMaterial(
          diffuse: ARKitMaterialProperty.color( Colors.blue.withOpacity(0.5)),
        ),
      ],
    );

    node = ARKitNode(
      geometry: plane,
      position: vector.Vector3(anchor.center.x, 0, anchor.center.z),
      rotation: vector.Vector4(1, 0, 0, -math.pi / 2),
    );

    controller.add(node!, parentNodeName: anchor.nodeName);
  }

void updateAnchor(ARKitAnchor anchor) {
  if (anchor is! ARKitPlaneAnchor) return;
  if (anchorId != anchor.identifier) return;

  // Remove old node
  if (node != null) {
    arkitController.remove(node!.name);
  }

  // Create new plane with updated dimensions
  final updatedPlane = ARKitPlane(
    width: anchor.extent.x,
    height: anchor.extent.z,
    materials: [
      ARKitMaterial(
        diffuse: ARKitMaterialProperty.color(Colors.blue.withOpacity(0.5)),

      ),
    ],
  );

  node = ARKitNode(
    geometry: updatedPlane,
    position: vector.Vector3(anchor.center.x, 0, anchor.center.z),
    rotation: vector.Vector4(1, 0, 0, -math.pi / 2),
  );

  arkitController.add(node!, parentNodeName: anchor.nodeName);
}

}
