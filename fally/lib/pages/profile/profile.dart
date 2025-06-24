import 'package:flutter/material.dart';

class ProfilePage extends StatelessWidget {
  const ProfilePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
      title: const Text('Profile'),
      centerTitle: true,
      ),
      body: Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: const [
        CircleAvatar(
          radius: 50,
          backgroundImage: AssetImage('assets/images/profile_avatar.png'),
        ),
        SizedBox(height: 20),
        Text(
          'John Doe',
          style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
        ),
        SizedBox(height: 8),
        Text(
          'johndoe@email.com',
          style: TextStyle(fontSize: 16, color: Colors.grey),
        ),
        ],
      ),
      ),
    );
  }
}
