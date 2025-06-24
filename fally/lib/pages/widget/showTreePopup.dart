import 'package:flutter/material.dart';
import '../tree/data.dart'; // Adjust the import based on your project structure
import './tree_assessement.dart';
void showTreePopup(BuildContext context, TreeData tree) {
  showDialog(
    context: context,
    builder: (context) => AlertDialog(
      title: Text('Tree #${tree.number}'),
      content: Text(tree.description),
      actions: [
        TextButton(
          onPressed: () {
            Navigator.pop(context);
            Navigator.pushNamed(context, '/report', arguments: tree);
          },
          child: Text('More Info'),
        ),
        TextButton(
          onPressed: () {
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => BranchAssessmentPage(),
              ),
            );
          },
          child: Text('Input Data'),
        ),
      ],
    ),
  );
}
