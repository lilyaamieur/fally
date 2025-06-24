import 'package:flutter/material.dart';
import 'package:google_maps_flutter/google_maps_flutter.dart';

class AddTreePage extends StatefulWidget {
  @override
  _AddTreePageState createState() => _AddTreePageState();
}

class _AddTreePageState extends State<AddTreePage> {
  GoogleMapController? _mapController;
  LatLng? _selectedLocation;
  Set<Marker> _markers = {};

  void _onMapTapped(LatLng position) {
    setState(() {
      _selectedLocation = position;
      _markers = {
        Marker(
          markerId: MarkerId('tree_marker'),
          position: position,
          infoWindow: InfoWindow(title: 'New Tree'),
        ),
      };
    });
  }

  void _addTree() {
    if (_selectedLocation != null) {
      // Here you can handle saving the tree location to your backend or state
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Tree added at: $_selectedLocation')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Add a Tree')),
      body: Stack(
        children: [
          Text("data")
        ],
      ),
    );
  }
}