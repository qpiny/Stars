name := """stars"""

version := "1.0"

scalaVersion := "2.11.5"

// Change this to another test framework if you prefer
libraryDependencies += "org.scalatest" %% "scalatest" % "2.2.4" % "test"

// Uncomment to use Akka
//libraryDependencies += "com.typesafe.akka" % "akka-actor_2.11" % "2.3.9"
//libraryDependencies += "org.jocl" % "jocl" % "0.1.9"
libraryDependencies ++= Seq(
	"org.jogamp.jocl" % "jocl" % "2.2.4",
	"org.jogamp.jocl" % "jocl-main" % "2.2.4",
	"org.jogamp.gluegen" % "gluegen" % "2.2.4",
	"org.jogamp.gluegen" % "gluegen-rt" % "2.2.4",
	"org.jogamp.gluegen" % "gluegen-rt-main" % "2.2.4")


mainClass in assembly := Some("org.rejna.stars.Main")