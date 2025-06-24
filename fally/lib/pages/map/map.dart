import 'package:flutter/material.dart';
import 'package:font_awesome_flutter/font_awesome_flutter.dart';
import '../widget/showTreePopup.dart';
import '../tree/data.dart';


class InteractiveMapPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Interactive Map'),
        centerTitle: true,
      ),
      body: Center(
        child: InteractiveViewer(
          maxScale: 5.0,
          minScale: 0.5,
          boundaryMargin: EdgeInsets.all(double.infinity),
          child: Stack(
            children: [
              // Map background with full size
              Image.asset('assets/images/map.jpg'),

              // Tree icons overlaid using absolute positioning inside same coordinate system
              ..._trees.map((tree) {
                return Positioned(
                  left: tree.position.dx,
                  top: tree.position.dy,
                  child: ClickableTreeIcon(
                      onTap: () => showTreePopup(context, tree),

                  ),
                );
              }).toList(),
            ],
          ),
        ),
      ),
    );
  }

  /// All your tree icon positions in one list (map coordinates)
  final List<TreeData> _trees = [
  TreeData(number: 1, position: Offset(170, 130), description: 'Oak tree near path'),
  TreeData(number: 2, position: Offset(170, 110), description: 'Young sapling'),
  TreeData(number: 3, position: Offset(170, 150), description: 'Maple tree'),
  TreeData(number: 4, position: Offset(170, 170), description: 'Healthy pine'),
  TreeData(number: 5, position: Offset(170, 190), description: 'Cedar tree'),
  TreeData(number: 6, position: Offset(170, 210), description: 'Shady tree spot'),
  TreeData(number: 7, position: Offset(170, 229), description: 'Marked for observation'),
  TreeData(number: 8, position: Offset(170, 250), description: 'Near bird nest'),
  TreeData(number: 9, position: Offset(185, 130), description: 'Fruit-bearing tree'),
  TreeData(number: 10, position: Offset(170, 110), description: 'Duplicate sapling'),
  TreeData(number: 11, position: Offset(185, 150), description: 'Freshly watered tree'),
  TreeData(number: 12, position: Offset(185, 170), description: 'Strong bark tree'),
  TreeData(number: 13, position: Offset(185, 190), description: 'Tree near bench'),
  TreeData(number: 14, position: Offset(185, 210), description: 'Tree with sign'),
  TreeData(number: 15, position: Offset(185, 229), description: 'Tree under care'),
  TreeData(number: 16, position: Offset(185, 250), description: 'Popular photo spot'),
  TreeData(number: 17, position: Offset(170, 270), description: 'Older tree species'),
  TreeData(number: 18, position: Offset(170, 290), description: 'Tree beside trail'),
  TreeData(number: 19, position: Offset(170, 310), description: 'Rustling tree'),
  TreeData(number: 20, position: Offset(170, 330), description: 'Low hanging branches'),
  TreeData(number: 21, position: Offset(185, 270), description: 'Hidden tree'),
  TreeData(number: 22, position: Offset(185, 290), description: 'Tree near flowers'),
  TreeData(number: 23, position: Offset(185, 310), description: 'Pine with cone'),
  TreeData(number: 24, position: Offset(185, 330), description: 'Tree near wall'),

  // Second cluster
  TreeData(number: 25, position: Offset(350, 130), description: 'Sunny tree'),
  TreeData(number: 26, position: Offset(350, 110), description: 'Bright foliage'),
  TreeData(number: 27, position: Offset(350, 150), description: 'Strong roots'),
  TreeData(number: 28, position: Offset(350, 170), description: 'Evergreen'),
  TreeData(number: 29, position: Offset(350, 190), description: 'Insect repellent test'),
  TreeData(number: 30, position: Offset(355, 210), description: 'Educational tag tree'),
  TreeData(number: 31, position: Offset(365, 229), description: 'Planted recently'),
  TreeData(number: 32, position: Offset(350, 250), description: 'Soil moisture monitored'),

  TreeData(number: 33, position: Offset(335, 130), description: 'Tree with shade'),
  TreeData(number: 34, position: Offset(335, 110), description: 'Branching wide'),
  TreeData(number: 35, position: Offset(335, 150), description: 'Low roots'),
  TreeData(number: 36, position: Offset(335, 170), description: 'Tree by water feature'),
  TreeData(number: 37, position: Offset(335, 190), description: 'Labelled for ID'),
  TreeData(number: 38, position: Offset(325, 210), description: 'Needs pruning'),
  TreeData(number: 39, position: Offset(315, 229), description: 'Tree by path curve'),
  TreeData(number: 40, position: Offset(335, 250), description: 'Well maintained'),

  TreeData(number: 41, position: Offset(350, 270), description: 'Tree near open field'),
  TreeData(number: 42, position: Offset(350, 290), description: 'Lush canopy'),
  TreeData(number: 43, position: Offset(350, 310), description: 'Bird nest observed'),
  TreeData(number: 44, position: Offset(350, 330), description: 'Healthy leaves'),

  TreeData(number: 45, position: Offset(335, 270), description: 'Protected tree'),
  TreeData(number: 46, position: Offset(335, 290), description: 'Tree near rock patch'),
  TreeData(number: 47, position: Offset(335, 310), description: 'Low maintenance'),
  TreeData(number: 48, position: Offset(335, 330), description: 'Last tree in cluster'),
];

}

/// ✅ Clickable icon with hover and scale feedback
class ClickableTreeIcon extends StatefulWidget {
  final VoidCallback onTap;

  const ClickableTreeIcon({Key? key, required this.onTap}) : super(key: key);

  @override
  State<ClickableTreeIcon> createState() => _ClickableTreeIconState();
}

class _ClickableTreeIconState extends State<ClickableTreeIcon> {
  bool _isHovered = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      cursor: SystemMouseCursors.click,
      onEnter: (_) => setState(() => _isHovered = true),
      onExit: (_) => setState(() => _isHovered = false),
      child: GestureDetector(
        onTap: widget.onTap,
        child: AnimatedScale(
          scale: _isHovered ? 1.3 : 1.0,
          duration: Duration(milliseconds: 150),
          child: FaIcon(
            FontAwesomeIcons.tree,
            size: 8,
            color: _isHovered ? Colors.lightGreen : Colors.green,
          ),
        ),
      ),
    );
  }
}
