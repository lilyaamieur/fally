import 'package:flutter/material.dart';
import 'dart:math';
import '../report/alert.dart';

class ToolsPage extends StatelessWidget {
  const ToolsPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Expert Tools')),
      body: ListView(children: const []),
      floatingActionButton: FloatingActionButton(
        child: const Icon(Icons.build),
        onPressed: () {
          showModalBottomSheet(
            context: context,
            builder: (context) => const ToolSelector(),
          );
        },
      ),
    );
  }
}

class ToolSelector extends StatelessWidget {
  const ToolSelector({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Wrap(
      children: [
        ListTile(
          leading: const Icon(Icons.straighten, color: Colors.green),
          title: const Text('Measurement Tool'),
          onTap: () {
            Navigator.pop(context);
            Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const MeasurementTool()),
            );
          },
        ),
        ListTile(
          leading: const Icon(Icons.rotate_90_degrees_ccw, color: Colors.green),
          title: const Text('Angle Verification'),
          onTap: () {
            Navigator.pop(context);
            Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const AngleVerificationTool()),
            );
          },
        ),
        /*ListTile(
          leading: const Icon(Icons.calculate, color: Colors.green),
          title: const Text('Expert Calculator'),
          onTap: () {
            Navigator.pop(context);
            Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const ExpertCalculator()),
            );
          },
        ),*/
        ListTile(
          leading: const Icon(Icons.warning, color: Colors.red),
          title: const Text('Report Issue'),
          onTap: () {
            Navigator.pop(context);
            Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const ReportIssuePage()),
            );
          },
        ),
      ],
    );
  }
}

class MeasurementTool extends StatefulWidget {
  const MeasurementTool({Key? key}) : super(key: key);

  @override
  State<MeasurementTool> createState() => _MeasurementToolState();
}

class _MeasurementToolState extends State<MeasurementTool> {
  double? length;
  double? width;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Measurement Tool')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              decoration: const InputDecoration(labelText: 'Length (cm)'),
              keyboardType: TextInputType.number,
              onChanged: (value) => setState(() {
                length = double.tryParse(value);
              }),
            ),
            TextField(
              decoration: const InputDecoration(labelText: 'Width (cm)'),
              keyboardType: TextInputType.number,
              onChanged: (value) => setState(() {
                width = double.tryParse(value);
              }),
            ),
            const SizedBox(height: 20),
            if (length != null && width != null)
              Text('Area: ${(length! * width!).toStringAsFixed(2)} cm²'),
          ],
        ),
      ),
    );
  }
}

class AngleVerificationTool extends StatefulWidget {
  const AngleVerificationTool({Key? key}) : super(key: key);

  @override
  State<AngleVerificationTool> createState() => _AngleVerificationToolState();
}

class _AngleVerificationToolState extends State<AngleVerificationTool> {
  double? sideA;
  double? sideB;
  double? sideC;
  double? angle;

  void calculateAngle() {
    if (sideA != null && sideB != null && sideC != null) {
      // Law of Cosines: cos(C) = (a² + b² - c²) / (2ab)
      double cosC =
          ((sideA! * sideA!) + (sideB! * sideB!) - (sideC! * sideC!)) /
          (2 * sideA! * sideB!);
      angle = acos(cosC) * 180 / pi;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Angle Verification')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Text("finding a difficulty to measure ?"),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                calculateAngle();
                setState(() {});
              },
              child: const Text('Open Measurement Tool'),
            ),
            TextField(
              decoration: const InputDecoration(labelText: 'Side A (cm)'),
              keyboardType: TextInputType.number,
              onChanged: (value) => setState(() {
                sideA = double.tryParse(value);
                calculateAngle();
              }),
            ),
            TextField(
              decoration: const InputDecoration(labelText: 'Side B (cm)'),
              keyboardType: TextInputType.number,
              onChanged: (value) => setState(() {
                sideB = double.tryParse(value);
                calculateAngle();
              }),
            ),
            TextField(
              decoration: const InputDecoration(labelText: 'Side C (cm)'),
              keyboardType: TextInputType.number,
              onChanged: (value) => setState(() {
                sideC = double.tryParse(value);
                calculateAngle();
              }),
            ),
            const SizedBox(height: 10),
            if (angle != null && !angle!.isNaN)
              Text('Angle (°): ${angle!.toStringAsFixed(2)}'),

            const SizedBox(height: 50),
            Text(
              "you want to verify the angle between sides A and B with side C using camera ?",
            ),

            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                calculateAngle();
                setState(() {});
              },
              child: const Text('Open Camera'),
            ),
          ],
        ),
      ),
    );
  }
}

/*
class ExpertCalculator extends StatefulWidget {
  const ExpertCalculator({Key? key}) : super(key: key);

  @override
  State<ExpertCalculator> createState() => _ExpertCalculatorState();
}

class _ExpertCalculatorState extends State<ExpertCalculator> {
  String expression = '';
  String result = '';

  void calculate() {
    try {
      // Simple parser for +, -, *, /
      result = _evaluateExpression(expression).toString();
    } catch (e) {
      result = 'Error';
    }
    setState(() {});
  }

  double _evaluateExpression(String expr) {
    // This is a very basic evaluator for demonstration.
    // For production, use a proper math parser package.
    expr = expr.replaceAll(' ', '');
    if (expr.contains('+')) {
      var parts = expr.split('+');
      return _evaluateExpression(parts[0]) + _evaluateExpression(parts[1]);
    }
    if (expr.contains('-')) {
      var parts = expr.split('-');
      return _evaluateExpression(parts[0]) - _evaluateExpression(parts[1]);
    }
    if (expr.contains('*')) {
      var parts = expr.split('*');
      return _evaluateExpression(parts[0]) * _evaluateExpression(parts[1]);
    }
    if (expr.contains('/')) {
      var parts = expr.split('/');
      return _evaluateExpression(parts[0]) / _evaluateExpression(parts[1]);
    }
    return double.parse(expr);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Expert Calculator')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              decoration: const InputDecoration(labelText: 'Expression'),
              onChanged: (value) => setState(() {
                expression = value;
              }),
            ),
            const SizedBox(height: 10),
            ElevatedButton(
              onPressed: calculate,
              child: const Text('Calculate'),
            ),
            const SizedBox(height: 20),
            Text('Result: $result'),
          ],
        ),
      ),
    );
  }
}

*/
