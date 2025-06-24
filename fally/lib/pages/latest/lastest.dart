import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../data/alerts_data.dart'; // ✅ Make sure this matches your file name exactly

class LatestNewsPage extends StatelessWidget {
  const LatestNewsPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
      title: const Text('Latest News'),
      centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(12.0),
        child: ListView.builder(
          itemCount: alertData.length,
          itemBuilder: (context, index) {
            final news = alertData[index];
            final String title = news['title'] ?? 'No Title';
            final String subtitle = news['subtitle'] ?? '';
            final String? date = news['date'];
            final String? treeNumber = news['treeNumber'];

            return Card(
              elevation: 3,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              margin: const EdgeInsets.symmetric(vertical: 10),
              child: ListTile(
                contentPadding: const EdgeInsets.symmetric(
                    horizontal: 20, vertical: 12),
                leading: const Icon(Icons.article_rounded,
                    color: Colors.lightGreen, size: 36),
                title: Text(
                  title,
                  style: GoogleFonts.poppins(
                    fontWeight: FontWeight.w600,
                    fontSize: 16,
                    color: Colors.black87,
                  ),
                ),
                subtitle: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      subtitle,
                      style: GoogleFonts.poppins(
                        fontSize: 14,
                        color: Colors.grey[700],
                      ),
                    ),
                    if (treeNumber != null) ...[
                      const SizedBox(height: 6),
                      Text(
                        '🌳 Tree #: $treeNumber',
                        style: GoogleFonts.poppins(
                          fontSize: 12,
                          color: Colors.teal[700],
                        ),
                      ),
                    ],
                    if (date != null) ...[
                      const SizedBox(height: 4),
                      Text(
                        '🗓 $date',
                        style: GoogleFonts.poppins(
                          fontSize: 12,
                          color: Colors.grey[500],
                        ),
                      ),
                    ],
                  ],
                ),
                trailing: const Icon(Icons.circle,
                    size: 16, color: Color.fromARGB(181, 158, 158, 158)),
                onTap: () {
                  // Optional: implement a detailed view
                },
              ),
            );
          },
        ),
      ),
    );
  }
}
