import 'package:flutter/material.dart';
import '../data/alerts_data.dart'; // Replace with actual path

class ReportIssuePage extends StatefulWidget {
  const ReportIssuePage({Key? key}) : super(key: key);

  @override
  State<ReportIssuePage> createState() => _ReportIssuePageState();
}

class _ReportIssuePageState extends State<ReportIssuePage> {
  final TextEditingController _titleController = TextEditingController();
  final TextEditingController _subtitleController = TextEditingController();
  final TextEditingController _treeNumberController = TextEditingController();

  void _submitAlert() {
    final title = _titleController.text.trim();
    final subtitle = _subtitleController.text.trim();
    final treeNumber = _treeNumberController.text.trim();

    if (title.isEmpty || subtitle.isEmpty || treeNumber.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Please fill in all fields')),
      );
      return;
    }

    final now = DateTime.now().toIso8601String().split('T').first;

    alertData.insert(0, {
      'title': title,
      'subtitle': subtitle,
      'date': now,
      'treeNumber': treeNumber,
    });

    Navigator.pop(context);
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Alert reported successfully!')),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Report an Issue')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            TextField(
              controller: _titleController,
              decoration: const InputDecoration(
                labelText: 'Issue Title',
                border: OutlineInputBorder(),
              ),
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _subtitleController,
              decoration: const InputDecoration(
                labelText: 'Issue Description',
                border: OutlineInputBorder(),
              ),
              maxLines: 3,
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _treeNumberController,
              decoration: const InputDecoration(
                labelText: 'Tree Number',
                border: OutlineInputBorder(),
              ),
              keyboardType: TextInputType.number,
            ),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              icon: const Icon(Icons.send),
              label: const Text('Submit Alert'),
              onPressed: _submitAlert,
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.redAccent,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
