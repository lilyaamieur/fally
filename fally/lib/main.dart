import 'package:flutter/material.dart';
import 'pages/measure/ar_measure.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ARKit Demo',
      theme: ThemeData(primarySwatch: Colors.blue),
      debugShowCheckedModeBanner: false,
      home: HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('ARKit Home')),
      body: Center(
        child: ElevatedButton(
          onPressed: () {
            // 👇 This is how you navigate to ArMeasur
            Navigator.push(
              context,
              MaterialPageRoute(builder: (context) => ArMeasur()),
            );
          },
          child: Text('Open AR Measure'),
        ),
      ),
    );
  }
}
