import 'package:flutter/material.dart';
import '../navigation/bottom_nav.dart';
import '../map/map.dart';
import '../profile/profile.dart';
import '../imaging/data_image.dart';
import '../_tools/tools.dart';
import '../latest/lastest.dart';
import '../add/add.dart';
// Assuming you have an imaging page

class HomePageWithNav extends StatefulWidget {
  const HomePageWithNav({Key? key}) : super(key: key);

  @override
  State<HomePageWithNav> createState() => _HomePageState();
}

class _HomePageState extends State<HomePageWithNav> {
  int _selectedIndex = 0;

  final List<Widget> _screens = [
    InteractiveMapPage(), // Replace with your actual home page widget
    LatestNewsPage(),
    AddTreePage(),
    const ProfilePage(), // Replace with your actual profile page widget
  ];

  void _onTabChanged(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _screens[_selectedIndex],
      bottomNavigationBar: CustomBottomNav(
        currentIndex: _selectedIndex,
        onTap: _onTabChanged,
      ),
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
    
    ;
  }
}
