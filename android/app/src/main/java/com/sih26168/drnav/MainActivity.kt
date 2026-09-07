package com.sih26168.drnav

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle

@OptIn(ExperimentalMaterial3Api::class)
class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { MaterialTheme(colorScheme = darkColorScheme()) { RecorderScreen() } }
    }

    @Composable
    private fun RecorderScreen() {
        val state by RecordingService.state.collectAsStateWithLifecycle()
        var mode by remember { mutableStateOf("vehicle") }
        var granted by remember {
            mutableStateOf(
                ContextCompat.checkSelfPermission(
                    this, Manifest.permission.ACCESS_FINE_LOCATION
                ) == PackageManager.PERMISSION_GRANTED
            )
        }

        val launcher = rememberLauncherForActivityResult(
            ActivityResultContracts.RequestMultiplePermissions()
        ) { result ->
            granted = result[Manifest.permission.ACCESS_FINE_LOCATION] == true
        }

        Column(
            Modifier.fillMaxSize().padding(20.dp).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(14.dp),
        ) {
            Text("Dead Reckoning Logger", fontSize = 24.sp, fontWeight = FontWeight.Bold)
            Text("SIH26168 · ISRO", fontSize = 13.sp, color = Color.Gray)

            if (!granted) {
                Card {
                    Column(Modifier.padding(16.dp)) {
                        Text("Location permission is required", fontWeight = FontWeight.SemiBold)
                        Spacer(Modifier.height(6.dp))
                        Text(
                            "Raw GNSS measurements — including NavIC — are only " +
                                "delivered to apps holding precise location access.",
                            fontSize = 13.sp,
                        )
                        Spacer(Modifier.height(10.dp))
                        Button(onClick = {
                            val perms = mutableListOf(
                                Manifest.permission.ACCESS_FINE_LOCATION,
                                Manifest.permission.ACCESS_COARSE_LOCATION,
                            )
                            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                                perms += Manifest.permission.POST_NOTIFICATIONS
                            }
                            launcher.launch(perms.toTypedArray())
                        }) { Text("Grant") }
                    }
                }
            }

            // -- mode ------------------------------------------------------
            Text("Mode", fontWeight = FontWeight.SemiBold)
            SingleChoiceSegmentedButtonRow(Modifier.fillMaxWidth()) {
                listOf("vehicle", "pedestrian").forEachIndexed { i, m ->
                    SegmentedButton(
                        selected = mode == m,
                        onClick = { mode = m },
                        enabled = !state.recording,
                        shape = SegmentedButtonDefaults.itemShape(i, 2),
                    ) { Text(m.replaceFirstChar { it.uppercase() }) }
                }
            }

            // -- the badge the whole project is about ----------------------
            SourceBadge(state)

            // -- live counters ---------------------------------------------
            Card {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween) {
                        Text("Elapsed"); Mono("%.1f s".format(state.elapsedS))
                    }
                    Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween) {
                        Text("IMU samples"); Mono("${state.imuRows}")
                    }
                    Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween) {
                        Text("GNSS fixes"); Mono("${state.gnssRows}")
                    }
                    HorizontalDivider(Modifier.padding(vertical = 4.dp))
                    Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween) {
                        Text("Satellites in fix")
                        Mono("${state.sky.satellitesInFix} / ${state.sky.satellitesVisible}")
                    }
                    Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween) {
                        Text("NavIC in fix", color = Color(0xFF7FD1B9))
                        Mono("${state.sky.navicInFix} / ${state.sky.navicVisible}")
                    }
                    Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween) {
                        Text("Mean C/N₀"); Mono("%.1f dB-Hz".format(state.sky.meanCn0))
                    }
                    Row(Modifier.fillMaxWidth(), Arrangement.SpaceBetween) {
                        Text("HDOP")
                        Mono(if (state.sky.hdop.isNaN()) "—" else "%.1f".format(state.sky.hdop))
                    }
                }
            }

            Button(
                onClick = {
                    val i = Intent(this@MainActivity, RecordingService::class.java).apply {
                        action = if (state.recording) RecordingService.ACTION_STOP
                                 else RecordingService.ACTION_START
                        putExtra(RecordingService.EXTRA_MODE, mode)
                    }
                    if (state.recording) startService(i) else startForegroundService(i)
                },
                enabled = granted,
                modifier = Modifier.fillMaxWidth().height(56.dp),
                colors = if (state.recording)
                    ButtonDefaults.buttonColors(containerColor = Color(0xFFB3261E))
                else ButtonDefaults.buttonColors(),
            ) { Text(if (state.recording) "Stop recording" else "Start recording", fontSize = 16.sp) }

            state.dir?.let {
                Text("Saving to", fontSize = 12.sp, color = Color.Gray)
                Text(it, fontSize = 11.sp, fontFamily = FontFamily.Monospace, color = Color.Gray)
            }
            if (state.message.isNotEmpty()) {
                Text(state.message, fontSize = 12.sp, color = Color.Gray)
            }
        }
    }

    /**
     * The demo's most important pixel: whether the position currently comes
     * from satellites or from dead reckoning, and for how long it has been the
     * latter. A judge reads this before they read any number on the screen.
     */
    @Composable
    private fun SourceBadge(state: RecorderState) {
        val age = state.secondsSinceFix
        val dead = state.recording && (age.isNaN() || age > 3.0)
        val bg = when {
            !state.recording -> Color(0xFF3A3A3A)
            dead -> Color(0xFFB88400)
            state.sky.navicInFix > 0 -> Color(0xFF1E7A5A)
            else -> Color(0xFF1E5A7A)
        }
        val label = when {
            !state.recording -> "IDLE"
            dead -> "DEAD RECKONING  %.0f s".format(if (age.isNaN()) 0.0 else age)
            state.sky.navicInFix > 0 -> "NavIC + GNSS FIX"
            else -> "GNSS FIX"
        }
        Box(
            Modifier.fillMaxWidth().background(bg, MaterialTheme.shapes.medium).padding(18.dp),
            contentAlignment = Alignment.Center,
        ) { Text(label, fontSize = 17.sp, fontWeight = FontWeight.Bold, color = Color.White) }
    }

    @Composable
    private fun Mono(text: String) =
        Text(text, fontFamily = FontFamily.Monospace, fontWeight = FontWeight.Medium)
}
