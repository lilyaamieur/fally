import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';

class ClickableTreeIcon extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Center(
      child: GestureDetector(
        onTap: () {
          // Action when icon is tapped
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Tree icon clicked!')),
          );
        },
        child: FaIcon(
          FontAwesomeIcons.tree,
          size: 10,
          color: Colors.green,
        ),
      ),
    );
  }
}
